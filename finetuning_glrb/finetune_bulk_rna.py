import os
from functools import partial
from os import path as osp
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    DefaultDataCollator,
)
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from datasets import load_dataset, load_from_disk
from sklearn.metrics import r2_score
from src.utils.train import get_logger
from caduceus.tokenization_caduceus import CaduceusTokenizer
from finetuning_glrb.utils import fsspec_exists, get_last_embedding_dimension

# Logger setup
log = get_logger(__name__)

# Constants for the upstream and downstream window sizes
WINDOW_SIZE_BP_UPSTREAM = 384
WINDOW_SIZE_BP_DOWNSTREAM = 256

def tokenize_variants(examples, tokenizer, max_length: int):
    """
    Tokenize sequence.

    Args:
        examples: A batch of items from the dataset.
        tokenizer: AutoTokenizer instance.
        max_length: Maximum length for tokenization.

    Returns:
        dict with tokenized input IDs.
    """
    ref_tokenized = tokenizer.batch_encode_plus(
        examples["sequence"],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )
    return {
        "ref_input_ids": ref_tokenized["input_ids"],
    }

def recast_chromosome(examples):
    """
    Recast chromosome to integer format.

    Returns:
        dict with chromosome recast as integers.
    """
    return {
        "chromosome": -1 if examples["chromosome"] == "X" else -2 if examples["chromosome"] == "Y" else int(examples["chromosome"])
    }

class MLP_BulkRNA(nn.Module):
    """
    Regression head for Bulk RNA prediction task.

    Args:
        input_size: Input size for the linear layer.
        hidden_size: Hidden layer size.
        output_size: Output size for the linear layer.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_BulkRNA, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sp1 = nn.Softplus()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.sp1(self.fc1(x)))

class DNAModelForBulkRNA(nn.Module):
    """
    DNA Model for Bulk RNA prediction.

    Args:
        args: Arguments containing model configurations.
    """
    def __init__(self, args):
        super().__init__()
        self.rcps = args.rcps
        self.bp_per_token = args.bp_per_token
        self.config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

        # Load the appropriate backbone model based on the model name
        if "nucleotide-transformer" in args.model_name.lower():
            self.backbone = AutoModelForMaskedLM.from_pretrained(args.model_name, trust_remote_code=True).esm
        else:
            self.backbone = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
        
        print(f"MODEL LOADED: {self.backbone}")
        self.inner_dim = get_last_embedding_dimension(self.backbone,self.rcps)
        print(f"Inner dim founded for the Foundation Model: {self.inner_dim}")
        self.head = MLP_BulkRNA(input_size=self.inner_dim, hidden_size=2*self.inner_dim, output_size=218)

    def forward(self, input_ids):
        # Get embeddings from the backbone
        embeds_out = self.backbone(input_ids)[0]
        num_channels = embeds_out.size(-1)
        
        # Calculate the window size and indexes
        window_size_upstream = WINDOW_SIZE_BP_UPSTREAM // self.bp_per_token
        window_size_downstream = WINDOW_SIZE_BP_DOWNSTREAM // self.bp_per_token
        start, end = -window_size_downstream, window_size_upstream

        if self.rcps:
            # If the model is RC-equivariant
            embeds = embeds_out[..., :num_channels // 2]
            expanded_indices = torch.arange(start, end, device=embeds.device).unsqueeze(0).expand(embeds.size(0), -1) + embeds.size(1) // 2
            expanded_indices = torch.clamp(expanded_indices, 0, embeds.size(1) - 1)

            # Extract the relevant window from the embeddings and average it through the sequence length dimension
            tokens_window_ref = torch.gather(
                embeds, 1,
                expanded_indices.unsqueeze(-1).expand(-1, -1, embeds.size(2))
            )
            tokens_window_ref = tokens_window_ref.mean(dim=1)

            #Same for the RC-equivalent
            rc_embeds = embeds_out[..., num_channels // 2:].contiguous().flip(dims=[1, 2])
            expanded_indices = torch.arange(start, end, device=rc_embeds.device).unsqueeze(0).expand(rc_embeds.size(0), -1) + rc_embeds.size(1) // 2
            expanded_indices = torch.clamp(expanded_indices, 0, rc_embeds.size(1) - 1)

            tokens_window_rc = torch.gather(
                rc_embeds, 1,
                expanded_indices.unsqueeze(-1).expand(-1, -1, rc_embeds.size(2))
            )
            tokens_window_rc = tokens_window_rc.mean(dim=1)

            #Combine the Reference and RC-Equivariant resulting embeddings
            aggregated_embeds = tokens_window_rc + tokens_window_ref
            return self.head(aggregated_embeds)

        else:
            # No reverse complement processing
            expanded_indices = torch.arange(start, end, device=embeds_out.device).unsqueeze(0).expand(embeds_out.size(0), -1) + embeds_out.size(1) // 2
            expanded_indices = torch.clamp(expanded_indices, 0, embeds_out.size(1) - 1)

            # Extract the relevant window
            tokens_window_ref = torch.gather(
                embeds_out, 1,
                expanded_indices.unsqueeze(-1).expand(-1, -1, embeds_out.size(2))
            )
            tokens_window_ref = tokens_window_ref.mean(dim=1)
            return self.head(tokens_window_ref)

class Lit_BulkRNAFinetuning(pl.LightningModule):
    """
    PyTorch Lightning model for fine-tuning on Bulk RNA prediction.

    Args:
        args: Arguments containing model and training configurations.
    """
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.setup()

    def setup(self,stage=None):
        self.rcps = self.hparams.rcps
        self.model = DNAModelForBulkRNA(self.hparams)
        self.criterion = nn.MSELoss()
        self.validation_step_preds = []
        self.validation_step_labels = []
        self.training_step_preds = []
        self.training_step_labels = []
        
    def forward(self, ref_input_ids):
        return self.model(ref_input_ids)

    def training_step(self, batch, batch_idx):
        ref_input_ids = batch["ref_input_ids"]
        labels = batch["labels"]

        logits = self.model(ref_input_ids)
        loss = self.criterion(logits, labels)
        self.log('train_loss', loss, on_epoch=True, on_step=True, sync_dist=True)
        
        # Track predictions and labels for R² score
        self.training_step_preds.extend(logits.detach().cpu().numpy())
        self.training_step_labels.extend(labels.detach().cpu().numpy())

        return loss
    
    def validation_step(self, batch, batch_idx):
        ref_input_ids = batch["ref_input_ids"]
        labels = batch["labels"]

        logits = self.model(ref_input_ids)
        loss = self.criterion(logits, labels)
        self.log('val_loss', loss, on_epoch=True, on_step=True, sync_dist=True)

        # Track predictions and labels for R² score
        self.validation_step_preds.extend(logits.detach().cpu().numpy())
        self.validation_step_labels.extend(labels.detach().cpu().numpy())

        return loss
    
    def on_validation_epoch_end(self):     
        # Calculate R² score for validation
        val_r2 = r2_score(self.validation_step_labels, self.validation_step_preds)
        self.log("validation/R2", val_r2, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_labels.clear()
        self.validation_step_preds.clear()
    
    def on_train_epoch_end(self):
        # Calculate R² score for training
        train_r2 = r2_score(self.training_step_labels, self.training_step_preds)
        self.log("train/R2", train_r2, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_labels.clear()
        self.training_step_preds.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

class BulkRNADataModule(pl.LightningDataModule):
    """
    Data module for Bulk RNA finetuning with PyTorch Lightning.

    Args:
        config: Configuration dictionary with data-related parameters.
    """
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.bp_per_token = config.bp_per_token
        self.model_name = config.model_name
        self.train_batch_size = config.train_batch_size
        self.test_batch_size = config.test_batch_size
        self.num_workers = config.num_workers
        self.train_ratio = config.train_ratio
        self.eval_ratio = config.eval_ratio
        self.cache_dir = "./"
        self.dataset = None

        # Initialize the tokenizer
        if "caduceus" in self.model_name:
            self.tokenizer = CaduceusTokenizer(
                model_max_length=self.seq_len,
                add_special_tokens=False
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    def prepare_data(self):
        # Download and preprocess data if not already done
        if not fsspec_exists(self._get_preprocessed_cache_file()):
            self._download_and_preprocess_data()

    def setup(self, stage=None):
        # Load the preprocessed dataset
        self.prepare_data()
        self.dataset = load_from_disk(self._get_preprocessed_cache_file())

        # Split the dataset into train and validation sets
        self.train_dataset = self.dataset["train"]
        self.val_dataset = self.dataset["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=DefaultDataCollator(return_tensors="pt"),
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            collate_fn=DefaultDataCollator(return_tensors="pt"),
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )

    def _get_preprocessed_cache_file(self):
        self.cache_dir = osp.join(
            "./", "data", "InstaDeepAI___genomics-long-range-benchmark",
            "variant_effect_pathogenic_clinvar", f"seqlen{self.seq_len}"
        )
        cache_file = os.path.join(self.cache_dir, "caduceus_char_token_preprocessed")
        return re.sub(r"=", "_", cache_file)

    def _download_and_preprocess_data(self):
        log.warning("Downloading and preprocessing data...")
        dataset = load_dataset(
            "InstaDeepAI/genomics-long-range-benchmark",
            task_name="bulk_rna_expression",
            sequence_length=self.seq_len,
            load_from_cache=False,
            trust_remote_code=True
        )
        try:
            del dataset["validation"]  # Remove empty validation split if it exists
        except KeyError:
            pass

        # Process data: filter sequences with too many 'N's, recast chromosomes, and tokenize
        dataset = dataset.filter(
            lambda example: example["sequence"].count('N') < 0.005 * self.seq_len,
            desc="Filter N's"
        )
        dataset = dataset.map(
            recast_chromosome,
            remove_columns=["chromosome"],
            desc="Recast chromosome"
        )
        dataset = dataset.map(
            partial(tokenize_variants, tokenizer=self.tokenizer, max_length=self.seq_len//self.bp_per_token),
            batch_size=1000,
            batched=True,
            remove_columns=["sequence"],
            desc="Tokenize",
            num_proc=self.num_workers
        )

        # Save processed dataset to disk
        dataset.save_to_disk(self._get_preprocessed_cache_file())
        log.warning("Data downloaded and preprocessed successfully.")

def finetune(args):
    """
    Main function to start the training process using PyTorch Lightning.

    Args:
        args: Command line arguments or configuration dictionary.
    """
    wandb.login(key=args.wandb_api_key)
    wandb_logger = WandbLogger(
        name=f"{args.name_wb}-{args.seq_len}",
        project="Bulk RNA Expression",
        log_model=True,
        save_dir=args.save_dir
    )
    data_module = BulkRNADataModule(args)
    data_module.setup()

    model = Lit_BulkRNAFinetuning(args)

    # Callbacks for early stopping and model checkpointing
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.save_dir}/checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    nb_device = "1" if "nucleotide-transformer" in args.model_name.lower() else "auto"

    # Set up the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        devices=nb_device,
        logger=wandb_logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        log_every_n_steps=1,
        limit_train_batches=args.train_ratio,
        limit_val_batches=args.eval_ratio,
        val_check_interval=args.log_interval,
        gradient_clip_val=1.0,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_sanity_val_steps=0
    )

    # Start the training process
    trainer.fit(model, data_module)

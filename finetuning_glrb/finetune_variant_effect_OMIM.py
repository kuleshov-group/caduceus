import os
from functools import partial
from os import path as osp
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer, AutoConfig
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from transformers import DefaultDataCollator
from src.utils.train import get_logger
from caduceus.tokenization_caduceus import CaduceusTokenizer
from finetuning_glrb.utils import fsspec_exists, get_last_embedding_dimension

# Logger setup
log = get_logger(__name__)

# Constants
WINDOW_SIZE_BP = 1536

def tokenize_variants(examples, tokenizer, max_length: int):
    """
    Tokenize reference and alternate sequences.

    Args:
        examples: A batch of items from the dataset.
        tokenizer: AutoTokenizer instance.
        max_length: Maximum length for tokenization.

    Returns:
        dict with tokenized input IDs for reference and alternate sequences.
    """
    ref_tokenized = tokenizer.batch_encode_plus(
        examples["ref_forward_sequence"],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )
    alt_tokenized = tokenizer.batch_encode_plus(
        examples["alt_forward_sequence"],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )
    return {
        "ref_input_ids": ref_tokenized["input_ids"],
        "alt_input_ids": alt_tokenized["input_ids"]
    }

def recast_chromosome(examples):
    """
    Recast chromosome to integer format.

    Returns:
        dict with recast chromosome as an integer.
    """
    return {
        "chromosome": -1 if examples["chromosome"] == "X" else -2 if examples["chromosome"] == "Y" else int(examples["chromosome"])
    }

def find_variant_idx(examples):
    """
    Find the index of the variant in the sequence.

    Args:
        examples: Items from the dataset (not batched).

    Returns:
        dict with the index of the variant.
    """
    idx = len(examples["ref_input_ids"]) // 2  # Assume variant is at the midpoint
    if examples["ref_input_ids"][idx] == examples["alt_input_ids"][idx]:
        idx = -1
        for i, (ref, alt) in enumerate(zip(examples["ref_input_ids"], examples["alt_input_ids"])):
            if ref != alt:
                idx = i
    return {"variant_idx": idx}

class MLP_VEP_OMIM(nn.Module):
    """
    MLP head for Variant Effect Prediction (OMIM).

    Args:
        input_size: Input size for the linear layer.
        hidden_size: Hidden layer size.
        output_size: Output size for the linear layer.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_VEP_OMIM, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sp1 = nn.Softplus()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.sp2 = nn.Softplus()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc3(self.sp2(self.fc2(self.sp1(self.fc1(x)))))

class DNAModelForOMIMFinetuning(nn.Module):
    """
    DNA Model for OMIM Variant Effect Prediction fine-tuning.

    Args:
        args: Arguments containing model configurations.
    """
    def __init__(self, args):
        super().__init__()
        self.rcps = args.rcps
        self.bp_per_token = args.bp_per_token
        self.config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

        if "nucleotide-transformer" in args.model_name.lower():
            self.backbone = AutoModelForMaskedLM.from_pretrained(args.model_name, trust_remote_code=True).esm
        else:
            self.backbone = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)

        print(f"MODEL LOADED: {self.backbone}")
        self.inner_dim = get_last_embedding_dimension(self.backbone,self.rcps)
        print(f"Inner dim found for the Foundation Model: {self.inner_dim}")
        self.head = MLP_VEP_OMIM(input_size=2*self.inner_dim, hidden_size=2*self.inner_dim, output_size=2)

    def forward(self, alt_input_ids, ref_input_ids, variant_idx):
        embeds_alternate = self.backbone(alt_input_ids)[0]
        embeds_reference = self.backbone(ref_input_ids)[0]
        window_size = WINDOW_SIZE_BP // self.bp_per_token // 2
        num_channels = embeds_alternate.size(-1)

        if self.rcps:
            embeds_alt = embeds_alternate[..., :num_channels // 2]
            embeds_ref = embeds_reference[..., :num_channels // 2]
            expanded_indices = torch.arange(-window_size, window_size + 1, device=variant_idx.device).unsqueeze(0) + variant_idx.unsqueeze(1)
            expanded_indices = torch.clamp(expanded_indices, 0, embeds_alt.size(1) - 1)

            windowed_embeds_alt = torch.gather(embeds_alt, 1, expanded_indices.unsqueeze(-1).expand(-1, -1, embeds_alt.size(2)))
            windowed_embeds_ref = torch.gather(embeds_ref, 1, expanded_indices.unsqueeze(-1).expand(-1, -1, embeds_ref.size(2)))

            mean_embeds_alt = windowed_embeds_alt.mean(dim=1)
            mean_embeds_ref = windowed_embeds_ref.mean(dim=1)

            concat_embeds = torch.cat([mean_embeds_alt, mean_embeds_ref], dim=-1)

            rc_embeds_alt = embeds_alternate[..., num_channels // 2:].contiguous().flip(dims=[1, 2])
            rc_embeds_ref = embeds_reference[..., num_channels // 2:].contiguous().flip(dims=[1, 2])

            rc_windowed_embeds_alt = torch.gather(rc_embeds_alt, 1, expanded_indices.unsqueeze(-1).expand(-1, -1, rc_embeds_alt.size(2)))
            rc_windowed_embeds_ref = torch.gather(rc_embeds_ref, 1, expanded_indices.unsqueeze(-1).expand(-1, -1, rc_embeds_ref.size(2)))

            rc_mean_embeds_alt = rc_windowed_embeds_alt.mean(dim=1)
            rc_mean_embeds_ref = rc_windowed_embeds_ref.mean(dim=1)

            rc_concat_embeds = torch.cat([rc_mean_embeds_alt, rc_mean_embeds_ref], dim=-1)

            final_window = concat_embeds + rc_concat_embeds
            return self.head(final_window)

        else:
            expanded_indices = torch.arange(-window_size, window_size + 1, device=variant_idx.device).unsqueeze(0) + variant_idx.unsqueeze(1)
            expanded_indices = torch.clamp(expanded_indices, 0, embeds_alternate.size(1) - 1)

            windowed_embeds_alt = torch.gather(embeds_alternate, 1, expanded_indices.unsqueeze(-1).expand(-1, -1, embeds_alternate.size(2))).mean(dim=1)
            windowed_embeds_ref = torch.gather(embeds_reference, 1, expanded_indices.unsqueeze(-1).expand(-1, -1, embeds_reference.size(2))).mean(dim=1)

            concat_embeds = torch.cat([windowed_embeds_alt, windowed_embeds_ref], dim=-1)
            return self.head(concat_embeds)

class Lit_OMIMFinetuning(pl.LightningModule):
    """
    PyTorch Lightning model for fine-tuning on OMIM Variant Effect Prediction.

    Args:
        args: Arguments containing model and training configurations.
    """
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.setup()

    def setup(self,stage=None):
        self.model = DNAModelForOMIMFinetuning(self.hparams)
        self.criterion = nn.CrossEntropyLoss()
        self.validation_step_preds = []
        self.validation_step_labels = []
        self.training_step_preds = []
        self.training_step_labels = []
        self.training_step_correct = 0
        self.training_step_total = 0
        self.validation_step_correct = 0
        self.validation_step_total = 0

    def forward(self, alt_input_ids, ref_input_ids, variant_idx):
        return self.model(alt_input_ids, ref_input_ids, variant_idx)

    def training_step(self, batch, batch_idx):
        ref_input_ids = batch["ref_input_ids"]
        alt_input_ids = batch["alt_input_ids"]
        variant_index = batch["variant_idx"]
        labels = batch["labels"]

        logits = self(alt_input_ids, ref_input_ids, variant_index)
        loss = self.criterion(logits, labels)
        self.log('train_loss', loss, on_epoch=True, on_step=True, sync_dist=True)

        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        self.training_step_correct += correct
        self.training_step_total += len(labels)

        all_labels = labels.cpu().numpy()
        all_predictions = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        self.training_step_preds.extend(all_predictions)
        self.training_step_labels.extend(all_labels)

        return loss

    def validation_step(self, batch, batch_idx):
        ref_input_ids = batch["ref_input_ids"]
        alt_input_ids = batch["alt_input_ids"]
        variant_index = batch["variant_idx"]
        labels = batch["labels"]

        logits = self(alt_input_ids, ref_input_ids, variant_index)
        loss = self.criterion(logits, labels)
        self.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True)

        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        self.validation_step_correct += correct
        self.validation_step_total += len(labels)

        all_labels = labels.cpu().numpy()
        all_predictions = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        self.validation_step_preds.extend(all_predictions)
        self.validation_step_labels.extend(all_labels)

    def on_validation_epoch_end(self):
        val_auroc = roc_auc_score(self.validation_step_labels, self.validation_step_preds)
        precision, recall, _ = precision_recall_curve(self.validation_step_labels, self.validation_step_preds)
        val_auprc = auc(recall, precision)
        val_accuracy = self.validation_step_correct / self.validation_step_total

        self.logger.experiment.log({
            "validation/AUROC": val_auroc,
            "validation/Accuracy": val_accuracy,
            "validation/AUPRC": val_auprc,
        })

        self.validation_step_labels.clear()
        self.validation_step_preds.clear()
        self.validation_step_correct = 0
        self.validation_step_total = 0

    def on_train_epoch_end(self):
        train_auroc = roc_auc_score(self.training_step_labels, self.training_step_preds)
        precision, recall, _ = precision_recall_curve(self.training_step_labels, self.training_step_preds)
        train_auprc = auc(recall, precision)
        train_accuracy = self.training_step_correct / self.training_step_total

        self.logger.experiment.log({
            "train/AUROC": train_auroc,
            "train/Accuracy": train_accuracy,
            "train/AUPRC": train_auprc
        })

        self.training_step_labels.clear()
        self.training_step_preds.clear()
        self.training_step_correct = 0
        self.training_step_total = 0

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

class VariantEffectPredictionDataModule(pl.LightningDataModule):
    """
    Data module for OMIM Variant Effect Prediction fine-tuning with PyTorch Lightning.

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

        if "caduceus" in self.model_name:
            self.tokenizer = CaduceusTokenizer(
                model_max_length=self.seq_len,
                add_special_tokens=False
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    def prepare_data(self):
        if not fsspec_exists(self._get_preprocessed_cache_file()):
            self._download_and_preprocess_data()

    def setup(self, stage=None):
        self.prepare_data()
        self.dataset = load_from_disk(self._get_preprocessed_cache_file())

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
            shuffle=True
        )

    def _get_preprocessed_cache_file(self):
        self.cache_dir = osp.join(
            "./", "data", "InstaDeepAI___genomics-long-range-benchmark",
            "variant_effect_pathogenic_omim", f"seqlen{self.seq_len}"
        )
        cache_file = os.path.join(self.cache_dir, "caduceus_char_token_preprocessed")
        return re.sub(r"=", "_", cache_file)

    def _download_and_preprocess_data(self):
        log.warning(f"Downloading and preprocessing data...")
        dataset = load_dataset(
            "InstaDeepAI/genomics-long-range-benchmark",
            task_name="variant_effect_pathogenic_omim",
            sequence_length=self.seq_len,
            load_from_cache=False,
            trust_remote_code=True
        )
        try:
            del dataset["validation"]
        except KeyError:
            pass

        dataset = dataset.filter(
            lambda example: example["ref_forward_sequence"].count('N') < 0.005 * self.seq_len,
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
            remove_columns=["ref_forward_sequence", "alt_forward_sequence"],
            desc="Tokenize",
            num_proc=self.num_workers
        )
        dataset = dataset.map(find_variant_idx, desc="Find variant idx")

        dataset.save_to_disk(self._get_preprocessed_cache_file())
        log.warning(f"Data downloaded and preprocessed successfully.")

def finetune(args):
    """
    Main function to start the training process for OMIM Variant Effect Prediction using PyTorch Lightning.

    Args:
        args: Command line arguments or configuration dictionary.
    """
    wandb.login(key=args.wandb_api_key)
    wandb_logger = WandbLogger(
        name=f"{args.name}-{args.seq_len}",
        project="Variant Effect Prediction OMIM",
        log_model=True
    )
    data_module = VariantEffectPredictionDataModule(args)
    data_module.setup()

    model = Lit_OMIMFinetuning(args)

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
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

    trainer.fit(model, data_module)

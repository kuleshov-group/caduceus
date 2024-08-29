import os
from functools import partial
from os import path as osp
import re
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer, AutoConfig
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from datasets import load_dataset, load_from_disk
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_recall_curve, auc, roc_auc_score
from transformers import DefaultDataCollator
from src.utils.train import get_logger
from caduceus.tokenization_caduceus import CaduceusTokenizer
from finetuning_glrb.utils import fsspec_exists, get_last_embedding_dimension

# Constants
WINDOW_SIZE_BP = 1536
DIST_TO_TSS = [[0, 30_000], [30_000, 100_000], [100_000, np.inf]]

# Logger setup
log = get_logger(__name__)

def recast_chromosome_tissue_dist2TSS(examples):
    """
    Recast chromosome to integer and retain tissue and distance to nearest TSS.

    Returns:
        dict with recast chromosome, tissue, and distance to TSS.
    """
    return {
        "chromosome": -1 if examples["chromosome"] in ["X","Y"] else int(examples["chromosome"]),
        "tissue": examples["tissue"],
        "distance_to_nearest_tss": examples["distance_to_nearest_tss"]
    }

def tokenize_variants(examples, tokenizer, max_length: int):
    """
    Tokenize sequences.

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

def dataset_tss_filter(data: Dataset, min_distance: int, max_distance: int):
    """
    Filter the data based on the distance to the nearest TSS.

    Args:
        data: Dataset to be filtered.
        min_distance: Minimum distance to the TSS.
        max_distance: Maximum distance to the TSS.

    Returns:
        Filtered dataset.
    """
    distance_mask = (data["distance_to_nearest_tss"] >= min_distance) & (data["distance_to_nearest_tss"] <= max_distance)
    filtered_data = {key: value[distance_mask] for key, value in data.items()}
    return filtered_data

class MLP_VEP(nn.Module):
    """
    MLP head for Variant Effect Prediction (VEP).

    Args:
        input_size: Input size for the linear layer.
        hidden_size: Hidden layer size.
        output_size: Output size for the linear layer.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_VEP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sp1 = nn.Softplus()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.sp2 = nn.Softplus()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc3(self.sp2(self.fc2(self.sp1(self.fc1(x)))))

class DNAModelForVEPFinetuning(nn.Module):
    """
    DNA Model for Variant Effect Prediction fine-tuning.

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
        self.head = MLP_VEP(input_size=2 * self.inner_dim + 1, hidden_size=2 * self.inner_dim, output_size=2)

    def forward(self, alt_input_ids, ref_input_ids, variant_idx, tissue_embed):
        # Get embeddings for the alternate and reference sequences
        embeds_alternate = self.backbone(alt_input_ids)[0]
        embeds_reference = self.backbone(ref_input_ids)[0]
        window_size = WINDOW_SIZE_BP // self.bp_per_token // 2
        num_channels = embeds_alternate.size(-1)

        if self.rcps:
            # Reverse complement processing
            embeds_alt = embeds_alternate[..., :num_channels // 2]
            embeds_ref = embeds_reference[..., :num_channels // 2]

            expanded_indices = torch.arange(-window_size, window_size + 1, device=variant_idx.device).unsqueeze(0) + variant_idx.unsqueeze(1)
            expanded_indices = torch.clamp(expanded_indices, 0, embeds_alt.size(1) - 1)

            # Extract windowed embeddings
            windowed_embeds_alt = torch.gather(embeds_alt, 1, expanded_indices.unsqueeze(-1).expand(-1, -1, embeds_alt.size(2)))
            windowed_embeds_ref = torch.gather(embeds_ref, 1, expanded_indices.unsqueeze(-1).expand(-1, -1, embeds_ref.size(2)))

            mean_embeds_alt = windowed_embeds_alt.mean(dim=1)
            mean_embeds_ref = windowed_embeds_ref.mean(dim=1)

            # Concatenate the embeddings
            concat_embeds = torch.cat([mean_embeds_alt, mean_embeds_ref, tissue_embed[..., None]], dim=-1)

            #Same for the RC-Equivalent part
            rc_embeds_alt = embeds_alternate[..., num_channels // 2:].contiguous().flip(dims=[1, 2])
            rc_embeds_ref = embeds_reference[..., num_channels // 2:].contiguous().flip(dims=[1, 2])

            rc_windowed_embeds_alt = torch.gather(rc_embeds_alt, 1, expanded_indices.unsqueeze(-1).expand(-1, -1, rc_embeds_alt.size(2)))
            rc_windowed_embeds_ref = torch.gather(rc_embeds_ref, 1, expanded_indices.unsqueeze(-1).expand(-1, -1, rc_embeds_ref.size(2)))

            rc_mean_embeds_alt = rc_windowed_embeds_alt.mean(dim=1)
            rc_mean_embeds_ref = rc_windowed_embeds_ref.mean(dim=1)

            rc_concat_embeds = torch.cat([rc_mean_embeds_alt, rc_mean_embeds_ref, tissue_embed[..., None]], dim=-1)

            final_window = concat_embeds + rc_concat_embeds
            return self.head(final_window)

        else:
            expanded_indices = torch.arange(-window_size, window_size + 1, device=variant_idx.device).unsqueeze(0) + variant_idx.unsqueeze(1)
            expanded_indices = torch.clamp(expanded_indices, 0, embeds_alternate.size(1) - 1)

            windowed_embeds_alt = torch.gather(embeds_alternate, 1, expanded_indices.unsqueeze(-1).expand(-1, -1, embeds_alternate.size(2))).mean(dim=1)
            windowed_embeds_ref = torch.gather(embeds_reference, 1, expanded_indices.unsqueeze(-1).expand(-1, -1, embeds_reference.size(2))).mean(dim=1)

            concat_embeds = torch.cat([windowed_embeds_alt, windowed_embeds_ref, tissue_embed[..., None]], dim=-1)
            return self.head(concat_embeds)

class LitVEPFinetuning(pl.LightningModule):
    """
    PyTorch Lightning model for fine-tuning on Variant Effect Prediction.

    Args:
        args: Arguments containing model and training configurations.
    """
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.setup()
        
    def setup(self,stage=None):
        self.model = DNAModelForVEPFinetuning(self.hparams)
        self.criterion = nn.CrossEntropyLoss()
        self.validation_step_preds = {i: [] for i in range(len(DIST_TO_TSS))}
        self.validation_step_labels = {i: [] for i in range(len(DIST_TO_TSS))}

    def forward(self, alt_input_ids, ref_input_ids, variant_idx, tissue_embed):
        return self.model(alt_input_ids, ref_input_ids, variant_idx, tissue_embed)

    def training_step(self, batch, batch_idx):
        ref_input_ids = batch["ref_input_ids"]
        alt_input_ids = batch["alt_input_ids"]
        variant_index = batch["variant_idx"]
        tissue_embed = batch["tissue_embed"]
        labels = batch["labels"]

        logits = self(alt_input_ids, ref_input_ids, variant_index, tissue_embed)
        loss = self.criterion(logits, labels)
        self.log('train_loss', loss, on_epoch=True, on_step=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        ref_input_ids = batch["ref_input_ids"]
        alt_input_ids = batch["alt_input_ids"]
        variant_index = batch["variant_idx"]
        tissue_embed = batch["tissue_embed"]
        labels = batch["labels"]
        distance_to_nearest_tss = batch["distance_to_nearest_tss"]

        logits = self(alt_input_ids, ref_input_ids, variant_index, tissue_embed)
        loss = self.criterion(logits, labels)
        self.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True)

        # Predictions for AUROC
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels_np = labels.cpu().numpy()

        for i, (min_dist, max_dist) in enumerate(DIST_TO_TSS):
            mask = ((distance_to_nearest_tss >= min_dist) & (distance_to_nearest_tss < max_dist)).cpu().numpy()
            filtered_preds = preds[mask]
            filtered_labels = labels_np[mask]

            if len(filtered_labels) > 0:
                self.validation_step_labels[i].extend(filtered_labels)
                self.validation_step_preds[i].extend(filtered_preds)

    def test_step(self, batch, batch_idx):
        ref_input_ids = batch["ref_input_ids"]
        alt_input_ids = batch["alt_input_ids"]
        variant_index = batch["variant_idx"]
        tissue_embed = batch["tissue_embed"]
        labels = batch["labels"]
        distance_to_nearest_tss = batch["distance_to_nearest_tss"]

        logits = self(alt_input_ids, ref_input_ids, variant_index, tissue_embed)

        # Predictions for AUROC
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels_np = labels.cpu().numpy()

        for i, (min_dist, max_dist) in enumerate(DIST_TO_TSS):
            mask = ((distance_to_nearest_tss >= min_dist) & (distance_to_nearest_tss < max_dist)).cpu().numpy()
            filtered_preds = preds[mask]
            filtered_labels = labels_np[mask]

            if len(filtered_labels) > 0:
                self.validation_step_labels[i].extend(filtered_labels)
                self.validation_step_preds[i].extend(filtered_preds)

    def on_validation_epoch_end(self):
        # Initialize lists to store all labels and predictions across all TSS distance buckets
        all_labels = []
        all_preds = []

        for i, (min_dist, max_dist) in enumerate(DIST_TO_TSS):
            if len(self.validation_step_labels[i]) > 0:
                val_auroc = roc_auc_score(self.validation_step_labels[i], self.validation_step_preds[i])
                precision, recall, _ = precision_recall_curve(self.validation_step_labels[i], self.validation_step_preds[i])
                val_auprc = auc(recall, precision)
                val_accuracy = accuracy_score(self.validation_step_labels[i], self.validation_step_preds[i])

                # Log metrics for each TSS distance bucket
                self.log(f'validation/TSS({min_dist}-{max_dist})/AUROC', val_auroc, on_epoch=True, sync_dist=True)
                self.log(f'validation/TSS({min_dist}-{max_dist})/Accuracy', val_accuracy, on_epoch=True, sync_dist=True)
                self.log(f'validation/TSS({min_dist}-{max_dist})/AUPRC', val_auprc, on_epoch=True, sync_dist=True)
                print(f'Bucket {i} [{min_dist}-{max_dist}] - AUROC: {val_auroc:.4f}')

                # Aggregate the labels and predictions
                all_labels.extend(self.validation_step_labels[i])
                all_preds.extend(self.validation_step_preds[i])

            self.validation_step_labels[i].clear()
            self.validation_step_preds[i].clear()
            
        # Compute overall metrics if there are any labels
        if all_labels:
            overall_auroc = roc_auc_score(all_labels, all_preds)
            precision, recall, _ = precision_recall_curve(all_labels, all_preds)
            overall_auprc = auc(recall, precision)
            overall_accuracy = accuracy_score(all_labels, all_preds)

            # Log overall metrics
            self.log('validation/overall/AUROC', overall_auroc, on_epoch=True, sync_dist=True)
            self.log('validation/overall/Accuracy', overall_accuracy, on_epoch=True, sync_dist=True)
            self.log('validation/overall/AUPRC', overall_auprc, on_epoch=True, sync_dist=True)
            print(f'Overall - AUROC: {overall_auroc:.4f}')
    

    def on_test_epoch_end(self):
        # Initialize lists to store all labels and predictions across all TSS distance buckets
        all_labels = []
        all_preds = []

        for i, (min_dist, max_dist) in enumerate(DIST_TO_TSS):
            if len(self.validation_step_labels[i]) > 0:
                val_auroc = roc_auc_score(self.validation_step_labels[i], self.validation_step_preds[i])
                precision, recall, _ = precision_recall_curve(self.validation_step_labels[i], self.validation_step_preds[i])
                val_auprc = auc(recall, precision)
                val_accuracy = accuracy_score(self.validation_step_labels[i], self.validation_step_preds[i])

                # Log metrics for each TSS distance bucket
                self.log(f'test/TSS({min_dist}-{max_dist})/AUROC', val_auroc, on_epoch=True, sync_dist=True)
                self.log(f'test/TSS({min_dist}-{max_dist})/Accuracy', val_accuracy, on_epoch=True, sync_dist=True)
                self.log(f'test/TSS({min_dist}-{max_dist})/AUPRC', val_auprc, on_epoch=True, sync_dist=True)
                print(f'Bucket {i} [{min_dist}-{max_dist}] - AUROC: {val_auroc:.4f}')

                # Aggregate the labels and predictions
                all_labels.extend(self.validation_step_labels[i])
                all_preds.extend(self.validation_step_preds[i])

            self.validation_step_labels[i].clear()
            self.validation_step_preds[i].clear()
            
        # Compute overall metrics if there are any labels
        if all_labels:
            overall_auroc = roc_auc_score(all_labels, all_preds)
            precision, recall, _ = precision_recall_curve(all_labels, all_preds)
            overall_auprc = auc(recall, precision)
            overall_accuracy = accuracy_score(all_labels, all_preds)

            # Log overall metrics
            self.log('test/overall/AUROC', overall_auroc, on_epoch=True, sync_dist=True)
            self.log('test/overall/Accuracy', overall_accuracy, on_epoch=True, sync_dist=True)
            self.log('test/overall/AUPRC', overall_auprc, on_epoch=True, sync_dist=True)
            print(f'Overall - AUROC: {overall_auroc:.4f}')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

class VariantEffectPredictionDataModule(pl.LightningDataModule):
    """
    Data module for Variant Effect Prediction finetuning with PyTorch Lightning.

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
        self.test_dataset = self.dataset["test"]

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
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            collate_fn=DefaultDataCollator(return_tensors="pt"),
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )
    
    def _split_dataset(self,selected_validation_chromosome):
        log.warning(f"SPLITTING THE DATASET INTO TRAIN AND VAL SET, VAL SET BEING CHROMOSOME {selected_validation_chromosome}")
        self.train_dataset = self.dataset["train"].filter(
            lambda example: example["chromosome"]
            != selected_validation_chromosome,
            keep_in_memory=True,
        )
        self.val_dataset = self.dataset["train"].filter(
            lambda example: example["chromosome"]
            == selected_validation_chromosome,
            keep_in_memory=True,
        )
        self.validation_chromosome = selected_validation_chromosome

    def _get_preprocessed_cache_file(self):
        cache_dir = osp.join(
            "./", "data", "InstaDeepAI___genomics-long-range-benchmark",
            "variant_effect_causal_eqtl", f"seqlen{self.seq_len}"
        )
        cache_file = os.path.join(cache_dir, "caduceus_char_token_preprocessed")
        return re.sub(r"=", "_", cache_file)

    def _download_and_preprocess_data(self):
        log.warning(f"Downloading and preprocessing data...")
        dataset = load_dataset(
            "InstaDeepAI/genomics-long-range-benchmark",
            task_name="variant_effect_causal_eqtl",
            sequence_length=self.seq_len,
            load_from_cache=False,
            trust_remote_code=True
        )
        try:
            del dataset["validation"]  # Remove empty validation split if it exists
        except KeyError:
            pass

        dataset = dataset.filter(
            lambda example: example["ref_forward_sequence"].count('N') < 0.005 * self.seq_len,
            desc="Filter N's"
        )
        dataset = dataset.map(
            recast_chromosome_tissue_dist2TSS,
            remove_columns=["chromosome", "tissue", "distance_to_nearest_tss"],
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

        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(dataset["test"]["tissue"])
        train_tissue_embed = label_encoder.transform(dataset["train"]["tissue"])
        dataset["train"] = dataset["train"].add_column("tissue_embed", train_tissue_embed)
        test_tissue_embed = label_encoder.transform(dataset["test"]["tissue"])
        dataset["test"] = dataset["test"].add_column("tissue_embed", test_tissue_embed)

        # Save to disk if running locally
        dataset.save_to_disk(self._get_preprocessed_cache_file())

        log.warning(f"Data downloaded and preprocessed successfully.")

def finetune(args):
    """
    Main function to start process for Variant Effect Prediction Finetuning eQTL using PyTorch Lightning.

    Args:
        args: Command line arguments or configuration dictionary.
    """
    wandb.login(key=args.wandb_api_key)
    data_module = VariantEffectPredictionDataModule(args)
    data_module.setup()

    np.random.seed(0)
    candidates = np.unique(data_module.dataset["train"]["chromosome"])
    held_chromosomes = np.random.choice(candidates,5,replace = False)

    for idx,val_chromosome in enumerate(held_chromosomes):

        wandb_logger = WandbLogger(
            name=f"{args.name_wb}-{args.seq_len}-fold-{idx+1}",
            project="Variant Effect Prediction Causal eQTL",
            log_model=True  # Automatically log model checkpoints
        )

        data_module._split_dataset(val_chromosome)
        model = LitVEPFinetuning(args)

        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{args.save_dir}/checkpoints",
            filename=f"best-checkpoint-on-chromosome{val_chromosome}",
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
            callbacks=[checkpoint_callback],
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
        trainer.test(model,data_module,ckpt_path=f"./{args.save_dir}/checkpoints/best-checkpoint-on-chromosome{val_chromosome}.ckpt")

        # Finish the current WandB run
        wandb.finish()

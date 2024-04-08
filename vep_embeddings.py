"""Dump model embeddings for VEP classification task.

"""

import argparse
import os
from functools import partial
from os import path as osp
from typing import Dict, Iterable, Optional

import enformer_pytorch
import fsspec
import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from sklearn import preprocessing
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, DefaultDataCollator

from src.dataloaders.utils.rc import string_reverse_complement
from src.utils.train import get_logger

WINDOW_SIZE_BP = 1536
log = get_logger(__name__)


class DNAEmbeddingModel(nn.Module):
    """Wrapper around HF model.

    Args:
        model_name_or_path: str, path to HF model.
    """
    def __init__(
            self,
            model_name_or_path: str,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        # Enformer uses different library for loading
        if "enformer" in model_name_or_path.lower():
            self.backbone = enformer_pytorch.from_pretrained(
                model_name_or_path,
                use_tf_gamma=False,
                use_checkpointing=True
            )
        # NT model is not compatible with AutoModel class
        elif "nucleotide-transformer" in model_name_or_path.lower():
            # NT LM `backbone` is under the `.esm` attribute
            self.backbone = AutoModelForMaskedLM.from_pretrained(model_name_or_path, trust_remote_code=True).esm
        else:
            self.backbone = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)

    def forward(self, input_ids):
        """Backbone forward pass to retrieve last_hidden_state."""
        if "enformer" in self.model_name_or_path.lower():
            # Enformer forward pass has different signature
            return self.backbone(input_ids, return_embeddings=True)[1]
        return self.backbone(input_ids).last_hidden_state

class EnformerTokenizer:
    """Enformer tokenizer."""
    # Order is important here! (See: https://github.com/lucidrains/enformer-pytorch?tab=readme-ov-file#usage)
    pad_token = "P"  # Padding token should be a character to avoid issues with tokenization
    encode_map = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, pad_token: -1}

    @classmethod
    def encode(
            cls, seq: str, max_length: Optional[int] = None, truncation: Optional[bool] = False
    ) -> Iterable[int]:
        """Convert bp to token ids."""
        if max_length is not None:
            assert max_length >= 0, "max_length should be a positive integer."
            if len(seq) < max_length:
                seq = seq + cls.pad_token * (max_length - len(seq))
            elif truncation:
                seq = seq[:max_length]
        return [cls.encode_map[bp] for bp in seq.upper()]

    @classmethod
    def batch_encode_plus(
            cls, seqs: Iterable[str], max_length: Optional[int] = None, truncation: Optional[bool] = False,
            **kwargs,  # ensures compatibility with HF tokenizer-like API
    ) -> Dict[str, Iterable[Iterable[int]]]:
        """Batch encode sequences using HF tokenizer-like API."""
        input_ids = [cls.encode(seq, max_length=max_length, truncation=truncation) for seq in seqs]
        return {"input_ids": input_ids}


def setup_distributed():
    """Set environment variables for distributed runs."""
    dist.init_process_group("nccl")


def cleanup_distributed():
    """Clean up processes from distributed runs."""
    dist.destroy_process_group()


def fsspec_exists(filename):
    """Check if file exists in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def fsspec_listdir(dirname):
    """Listdir in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    return fs.ls(dirname)


# Processing functions
def recast_chromosome_tissue_dist2TSS(examples):
    """Recast chromosome to int."""
    return {
        "chromosome": -1 if examples["chromosome"] == "X" else int(examples["chromosome"]),
        "tissue": examples["tissue"],
        "distance_to_nearest_tss": examples["distance_to_nearest_tss"]
    }


def tokenize_variants(examples, tokenizer, max_length: int):
    """Tokenize sequence.

    Args:
        examples: (batch of) items from the dataset.
        tokenizer: AutoTokenizer.
        max_length: int.
    Returns:
        dict with values as list of token ids.
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
    ref_rc_tokenized = tokenizer.batch_encode_plus(
        [string_reverse_complement(seq) for seq in examples["ref_forward_sequence"]],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )
    alt_rc_tokenized = tokenizer.batch_encode_plus(
        [string_reverse_complement(seq) for seq in examples["alt_forward_sequence"]],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )

    return {
        "ref_input_ids": ref_tokenized["input_ids"],
        "alt_input_ids": alt_tokenized["input_ids"],
        "ref_rc_input_ids": ref_rc_tokenized["input_ids"],
        "alt_rc_input_ids": alt_rc_tokenized["input_ids"],
    }


def find_variant_idx(examples):
    """Find token location that differs between reference and variant sequence.

    Args:
        examples: items from the dataset (not batched).
    Returns:
        dict with values index of difference.
    """
    # Guess that variant is at halfway point
    idx = len(examples["ref_input_ids"]) // 2
    if examples["ref_input_ids"][idx] == examples["alt_input_ids"][idx]:
        # If no, loop through sequence and find variant location
        idx = -1
        for i, (ref, alt) in enumerate(zip(examples["ref_input_ids"], examples["alt_input_ids"])):
            if ref != alt:
                idx = i
    # Same as above, but for reverse complement
    rc_idx = len(examples["ref_rc_input_ids"]) // 2 - 1
    if examples["ref_rc_input_ids"][rc_idx] == examples["alt_rc_input_ids"][rc_idx]:
        rc_idx = -1
        for i, (ref, alt) in enumerate(zip(examples["ref_rc_input_ids"], examples["alt_rc_input_ids"])):
            if ref != alt:
                rc_idx = i
    return {"variant_idx": idx, "rc_variant_idx": rc_idx}


def prepare_dataset(args, tokenizer):
    """Prepare or load the tokenized dataset."""
    # Data Preprocessing
    num_tokens = args.seq_len // args.bp_per_token

    # Load data
    cache_dir = osp.join(
        os.getenv("HF_HOME"), "datasets", "InstaDeepAI___genomics-long-range-benchmark",
        "variant_effect_gene_expression", f"seqlen={args.seq_len}"
    )
    if "nucleotide-transformer" in args.model_name_or_path.lower():  # NT uses 6-mers, so tokenization is different
        preprocessed_cache_file = osp.join(cache_dir, "6mer_token_preprocessed")

    elif "enformer" in args.model_name_or_path.lower():
        # Enformer tokenization requires having vocab of just `A,C,G,T,N` (in that order)
        preprocessed_cache_file = osp.join(cache_dir, "enformer_char_token_preprocessed")
    else:
        preprocessed_cache_file = osp.join(cache_dir, "char_token_preprocessed")
    log.warning(f"Cache dir: {cache_dir}")
    log.warning(f"Cache dir preprocessed: {preprocessed_cache_file}")

    if not fsspec_exists(preprocessed_cache_file):
        if dist.get_rank() == 0:
            dataset = load_dataset(
                "InstaDeepAI/genomics-long-range-benchmark",
                task_name="variant_effect_gene_expression",
                sequence_length=args.seq_len,
                load_from_cache=False,
            )
            log.warning("Dataset loaded. Cached to disk:")
            log.warning(osp.dirname(list(dataset.cache_files.values())[0][0]["filename"]))
            try:
                del dataset["validation"]  # `validation` split is empty
            except KeyError:
                pass

            # Process data
            dataset = dataset.filter(
                lambda example: example["ref_forward_sequence"].count('N') < 0.005 * args.seq_len,
                desc="Filter N's"
            )
            dataset = dataset.map(
                recast_chromosome_tissue_dist2TSS,
                remove_columns=["chromosome", "tissue", "distance_to_nearest_tss"],
                desc="Recast chromosome"
            )
            dataset = dataset.map(
                partial(tokenize_variants, tokenizer=tokenizer, max_length=num_tokens),
                batch_size=1000,
                batched=True,
                remove_columns=["ref_forward_sequence", "alt_forward_sequence"],
                desc="Tokenize"
            )
            dataset = dataset.map(find_variant_idx, desc="Find variant idx")
            dataset.save_to_disk(preprocessed_cache_file)
    dist.barrier()  # Processes need to wait for dataset to be saved to disk (if not already done)
    dataset = load_from_disk(preprocessed_cache_file)
    log.warning(f"Loaded preprocessed dataset from {preprocessed_cache_file}")
    log.warning(dataset)
    return dataset


def get_backbone_model(args, device):
    """Get the backbone model."""

    model = DNAEmbeddingModel(
        model_name_or_path=args.model_name_or_path,
    )
    model.eval()
    return DDP(model.to(device))


def concat_storage_dict_values(storage_dict):
    """Helper method that combines lists of tensors in storage_dict into a single torch.Tensor."""
    return {key: torch.cat(storage_dict[key], dim=0) for key in storage_dict.keys()}


def dump_embeddings(args, dataset, model, device):
    """Dump embeddings to disk."""
    def extract_embeddings(item_ref, item_alt, variant_idx):
        """Extract embedding representation from last layer outputs

        Args:
            item_ref: torch.Tensor, shape (batch_size, seq_len, hidden_size) Ref embedding
            item_alt: torch.Tensor, shape (batch_size, seq_len, hidden_size) Alt embedding
            variant_idx: torch.Tensor, shape (batch_size,) Index of variant
        Returns:
            layer_metrics: dict, with values to save to disk
        """
        layer_metrics = {}

        # Compute windowed statistics
        if "enformer" in args.model_name_or_path.lower():
            window_size = WINDOW_SIZE_BP // 128  # Enformer's receptive field is 128
            # We also need to override variant_idx since Enformer model reduces to target_length of 896
            variant_idx = torch.ones_like(variant_idx) * item_ref.size(1) // 2
        else:
            window_size = WINDOW_SIZE_BP // args.bp_per_token

        # Add 1 so that window is: [window // 2 - SNP - window // 2]
        start, end = -window_size // 2, window_size // 2 + 1
        expanded_indices = torch.arange(start, end, device=item_ref.device).unsqueeze(0) + \
                           variant_idx.unsqueeze(1).to(item_ref.device)
        expanded_indices = torch.clamp(expanded_indices, 0, item_ref.size(1) - 1)  # Handle boundary conditions
        tokens_window_ref = torch.gather(
            item_ref, 1,
            expanded_indices.unsqueeze(-1).expand(-1, -1, item_ref.size(2))
        ).mean(dim=1)
        tokens_window_alt = torch.gather(
            item_alt, 1,
            expanded_indices.unsqueeze(-1).expand(-1, -1, item_ref.size(2))
        ).mean(dim=1)
        layer_metrics["concat_avg_ws"] = torch.cat([tokens_window_ref, tokens_window_alt], dim=-1)
        return layer_metrics

    embeds_path = osp.join(args.downstream_save_dir, args.name)
    os.makedirs(embeds_path, exist_ok=True)

    dataloader_params = {
        "batch_size": args.embed_dump_batch_size,
        "collate_fn": DefaultDataCollator(return_tensors="pt"),
        "num_workers": args.num_workers,
        "pin_memory": False,
        "shuffle": False,
        "drop_last": True
    }

    # Process label_encoder = preprocessing.LabelEncoder()
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(dataset["test"]["tissue"])
    train_tissue_embed = label_encoder.transform(dataset["train"]["tissue"])
    dataset["train"] = dataset["train"].add_column("tissue_embed", train_tissue_embed)
    test_tissue_embed = label_encoder.transform(dataset["test"]["tissue"])
    dataset["test"] = dataset["test"].add_column("tissue_embed", test_tissue_embed)

    if not all([
        fsspec_exists(osp.join(embeds_path, f"{split_name}_embeds_combined.pt")) for split_name in dataset.keys()
    ]):
        for split_name, split in dataset.items():
            sampler = DistributedSampler(
                split,
                shuffle=dataloader_params.get("shuffle", False),
                drop_last=dataloader_params.get("drop_last", True),
            )

            dl = DataLoader(split, **dataloader_params, sampler=sampler)

            storage_dict = {
                "concat_avg_ws": [],
                "rc_concat_avg_ws": [],
                "chromosome": [],
                "labels": [],
                "distance_to_nearest_tss": [],
                "tissue_embed": [],
            }

            with torch.no_grad():

                for batch_idx, batch in tqdm(
                        enumerate(dl), total=len(dl), desc=f"[RANK {dist.get_rank()}] Embedding {split_name}",
                        disable=dist.get_rank() != 0  # Only rank 0 updates pbar
                ):
                    for key in ["chromosome", "labels", "distance_to_nearest_tss", "tissue_embed"]:
                        storage_dict[key].append(batch[key].to("cpu", non_blocking=True))
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output_alt = model(batch["alt_input_ids"].to(device))
                        output_ref = model(batch["ref_input_ids"].to(device))
                        if args.rcps:
                            num_channels = output_alt.size(-1)
                            # Flip along length and channel dims to preserve RC equivariance
                            # i.e. output_rc(RC(inputs)) = outputs(inputs)
                            output_alt_rc = output_alt[..., num_channels // 2:].contiguous().flip(dims=[1, 2])
                            output_ref_rc = output_ref[..., num_channels // 2:].contiguous().flip(dims=[1, 2])
                            output_alt = output_alt[..., :num_channels // 2]
                            output_ref = output_ref[..., :num_channels // 2]

                        else:
                            # Flip along length dim so variant_idx aligns
                            output_alt_rc = model(batch["alt_rc_input_ids"].to(device)).contiguous().flip(dims=[1])
                            output_ref_rc = model(batch["ref_rc_input_ids"].to(device)).contiguous().flip(dims=[1])

                    metrics = extract_embeddings(
                        item_ref=output_ref,
                        item_alt=output_alt,
                        variant_idx=batch["variant_idx"],
                    )
                    for key, value in metrics.items():
                        storage_dict[key].append(metrics[key].to("cpu", non_blocking=True))

                    metrics_rc = extract_embeddings(
                        item_ref=output_ref_rc,
                        item_alt=output_alt_rc,
                        variant_idx=batch["variant_idx"],
                    )
                    for key, value in metrics_rc.items():
                        storage_dict[f"rc_{key}"].append(metrics_rc[key].to("cpu", non_blocking=True))

                    if batch_idx % 100 == 0:
                        # Every machine should print progress updates
                        print(f"[RANK {dist.get_rank()}] Completed index: {batch_idx}/{len(dl)}")

                storage_dict_temp = concat_storage_dict_values(storage_dict)
                with fsspec.open(osp.join(embeds_path, f"{split_name}_embeds_{dist.get_rank()}.pt"), "wb") as f:
                    torch.save(storage_dict_temp, f)
                print(f"[RANK {dist.get_rank()}] Saved {split_name} to {osp.join(embeds_path, f'{split_name}_embeds_{dist.get_rank()}.pt')}")
    else:
        log.warning("Embeddings already exist, skipping!")


def combine_embeddings(embeds_path):
    """Combine embeddings from different files."""
    # Check if combined embeddings exist, and if not, aggregate them
    for split in ["train", "test"]:
        if not fsspec_exists(osp.join(embeds_path, f"{split}_embeds_combined.pt")):
            storage_dict = {
                "concat_avg_ws": [],
                "rc_concat_avg_ws": [],
                "chromosome": [],
                "labels": [],
                "distance_to_nearest_tss": [],
                "tissue_embed": [],
            }
            for filename in fsspec_listdir(embeds_path):
                if f"{split}_embeds_" in filename:
                    log.warning(f"Loading data from: {filename}")
                    with fsspec.open(filename, "rb") as f:
                        tmp_data = torch.load(f)
                    for key in storage_dict.keys():
                        storage_dict[key].append(tmp_data[key])
            storage_dict = concat_storage_dict_values(storage_dict)
            log.warning(f"Saving combined data to: {embeds_path}/{split}_embeds_combined.pt")
            with fsspec.open(osp.join(embeds_path, f"{split}_embeds_combined.pt"), "wb") as f:
                torch.save(storage_dict, f)


def main(args):
    """Main entry point."""
    # Reproducibility
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    # Init distributed
    log.warning("Initializing distributed...")
    dist.init_process_group("nccl")
    print(f"[RANK {dist.get_rank()}] Distributed initialized: rank {dist.get_rank()}")  # All processes print this
    # Setup device
    device = torch.device(f"cuda:{dist.get_rank()}")
    print(f"[RANK {dist.get_rank()}] Using device: {device}.")  # All processes print this

    # Init tokenizer
    if "enformer" in args.model_name_or_path.lower():
        # Enformer tokenization requires having vocab of just `A,C,G,T,N` (in that order)
        tokenizer = EnformerTokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Get dataset
    dist.barrier()
    dataset = prepare_dataset(args, tokenizer)

    # Get model
    dist.barrier()
    model = get_backbone_model(args, device)
    log.warning("Model loaded.")

    # Dump embeddings
    dist.barrier()
    dump_embeddings(args, dataset, model, device)

    # Combine embeddings into single file
    dist.barrier()
    cleanup_distributed()
    combine_embeddings(osp.join(args.downstream_save_dir, args.name))


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seq_len", type=int, default=131072,
                        help="Sequence length (in bp)..")
    parser.add_argument("--bp_per_token", type=int, default=1,
                        help="Number of base pairs per token.")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--downstream_save_dir", type=str, default="./outputs/downstream/vep_embeddings",
                        help="Directory to save downstream task.")
    parser.add_argument("--name", type=str, default=None, help="Embeddings model name.")
    parser.add_argument("--rcps", default=False, action="store_true", help="Use RCPS.")
    parser.add_argument("--no-rcps", dest="rcps", action="store_false", help="Do not use RCPS.")
    parser.add_argument("--embed_dump_batch_size", type=int, default=1,
                        help="Batch size for embedding dump.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers.")
    opts, _ = parser.parse_known_args()
    log.warning("*** Args ************************")
    for k, v in vars(opts).items():
        log.warning(f"  - {k}: {v}")
    log.warning("******************************\n")

    main(opts)


import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from os import path as osp
import argparse
import torch


from src.utils.train import get_logger
from finetuning_glrb.finetune_variant_effect_pathogenic_clinvar import finetune as finetune_vep_clinvar
from finetuning_glrb.finetune_variant_effect_OMIM import finetune as main_omim
from finetuning_glrb.finetune_variant_effect_causal_eqtl import finetune as finetune_vep_eqtl
from finetuning_glrb.finetune_bulk_rna import finetune as finetune_bulk_rna_expression
from finetuning_glrb.finetune_chromatin import finetune_histone_marks,finetune_dna_accessibility
from finetuning_glrb.finetune_regulatory_elements import finetune_enhancers, finetune_promoters
from finetuning_glrb.finetune_cage import finetune as finetune_cage

log = get_logger(__name__)

def main(opts):
    # Check if the value of args.task matches one of the predefined options
    if opts.task == "variant_effect_causal_eqtl":
        finetune_vep_eqtl(opts)
    elif opts.task == "variant_effect_pathogenic_clinvar":
        finetune_vep_clinvar(opts)
    elif opts.task == "variant_effect_pathogenic_omim":
        main_omim(opts)
    elif opts.task == "bulk_rna_expression":
        finetune_bulk_rna_expression(opts)
    elif opts.task == "cage_prediction":
        finetune_cage(opts)
    elif opts.task == "chromatin_features_histone_marks":
        finetune_histone_marks(opts)   
    elif opts.task == "chromatin_features_dna_accessibility":
        finetune_dna_accessibility(opts)    
    elif opts.task == "regulatory_element_promoter":
        finetune_promoters(opts)  
    elif opts.task == "regulatory_element_enhancer":
        finetune_enhancers(opts)
        



if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "variant_effect_causal_eqtl",
            "variant_effect_pathogenic_clinvar",
            "variant_effect_pathogenic_omim",
            "cage_prediction",
            "bulk_rna_expression",
            "chromatin_features_histone_marks",
            "chromatin_features_dna_accessibility",
            "regulatory_element_promoter",
            "regulatory_element_enhancer"
        ],
        required=True,
        help="Choose one of the predefined variant effects."
    )
    parser.add_argument("--seq_len", type=int, default=131072,
                        help="Sequence length (in bp)..")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model to fine-tune")
    parser.add_argument("--bp_per_token", type = int, default = 1, help = "Number of baise pairs per token.")
    parser.add_argument("--save_dir", type=str, default="./outputs/downstream/vep_embeddings",
                        help="Directory to save downstream task.")
    parser.add_argument("--wandb_api_key",type=str,default=None,help="Weights & Biases API key for logging.")
    parser.add_argument("--name_wb", type=str, default=None, help="Embeddings model name.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Batch size for testing/validation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers.")
    parser.add_argument("--rcps", type=bool, default=False, help="Using rcps when extracting embeddings or not.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train.")
    parser.add_argument(
        "--precision",
        type=str,
        choices=['transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true'],
        required=True,
        help="Choose one mode."
    )
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                        help="Learning rate for optimizer")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--log_interval", type=int, default=5, 
                        help="Log interval")
    parser.add_argument("--train_ratio", type=float, default=1.0, 
                        help="Evaluation data ratio")
    parser.add_argument("--eval_ratio", type=float, default=1.0, 
                        help="Evaluation data ratio")
    opts = parser.parse_args()
    log.warning("*** Args ************************")
    for k, v in vars(opts).items():
        log.warning(f"  - {k}: {v}")
    log.warning("******************************\n")

    main(opts)
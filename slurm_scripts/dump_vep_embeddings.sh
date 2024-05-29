#!/bin/bash
#SBATCH --get-user-env                      # Retrieve the users login environment
#SBATCH -t 96:00:00                         # Time limit (hh:mm:ss)
#SBATCH --mem=100G                          # RAM
#SBATCH --gres=gpu:8                        # Number of GPUs
#SBATCH --ntasks-per-node=8                 # Should correspond to num devices (at least 1-1 task to GPU)
##SBATCH --cpus-per-task=4                  # Number of CPU cores per task
#SBATCH -N 1                                # Number of nodes
#SBATCH --requeue                           # Requeue job if it fails
#SBATCH --job-name=vep_embed                # Job name
#SBATCH--output=../watch_folder/%x_%j.log   # Output file name
#SBATCH --open-mode=append                  # Do not overwrite logs

NUM_WORKERS=2
NUM_DEVICES=8

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh
export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # Needed for setting deterministic functions for reproducibility

#####################################################################################
# Choose from one of the following:

## Enformer
#seq_len=196608
#bp_per_token=1
#embed_dump_batch_size=1
#model_name_or_path="EleutherAI/enformer-official-rough"
#name="enformer-seqlen=196k"
#rcps_flag="no-rcps"

## NTv2
#seq_len=12288  # 2048 (seq len) * 6 (kmers)
#bp_per_token=6
#embed_dump_batch_size=1
#model_name_or_path="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
#name="NTv2_downstream-seqlen=12k"
#rcps_flag="no-rcps"

## Hyena
#seq_len=131072
#bp_per_token=1
#embed_dump_batch_size=1
#model_name_or_path="LongSafari/hyenadna-medium-160k-seqlen-hf"
#name="hyena_downstream-seqlen=131k"
#rcps_flag="no-rcps"

## Caduceus-Ph
#seq_len=131072
#bp_per_token=1
#embed_dump_batch_size=1
#model_name_or_path="kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
#name="caduceus-ph_downstream-seqlen=131k"
#rcps_flag="no-rcps"

## Caduceus-PS
#seq_len=131072
#bp_per_token=1
#embed_dump_batch_size=1
#model_name_or_path="kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
#name="caduceus-ps_downstream-seqlen=131k"
#rcps_flag="rcps"
#####################################################################################

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=${NUM_DEVICES} \
    vep_embeddings.py \
      --num_workers=${NUM_WORKERS} \
      --seq_len=${seq_len}  \
      --bp_per_token=${bp_per_token}  \
      --embed_dump_batch_size=${embed_dump_batch_size} \
      --name="${name}"  \
      --model_name_or_path="${model_name_or_path}" \
      --"${rcps_flag}"

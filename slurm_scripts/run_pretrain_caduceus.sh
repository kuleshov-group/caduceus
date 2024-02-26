#!/bin/bash
#SBATCH --get-user-env                      # Retrieve the users login environment
#SBATCH -t 96:00:00                         # Time limit (hh:mm:ss)
#SBATCH --mem=100G                          # RAM
#SBATCH --gres=gpu:8                        # Number of GPUs
#SBATCH --ntasks-per-node=8                 # Should correspond to num devices (at least 1-1 task to GPU)
##SBATCH --cpus-per-task=4                   # Number of CPU cores per task
#SBATCH -N 1                                # Number of nodes
#SBATCH --requeue                           # Requeue job if it fails
#SBATCH --job-name=caduceus_ps              # Job name
#SBATCH --output=../watch_folder/%x_%j.log  # Log file
#SBATCH --open-mode=append                  # Do not overwrite logs

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh
export HYDRA_FULL_ERROR=1

NUM_DEVICES=8

# Run script
D_MODEL=256
N_LAYER=16
LR="8e-3"
BIDIRECTIONAL_STRATEGY="add"
BIDIRECTIONAL_WEIGHT_TIE="true"
RCPS="true"
RC_AUG="false"
MODEL_NAME="caduceus"
WANDB_NAME="caduceus-ps_seqlen-131k_d_model-${D_MODEL}_n_layer-${N_LAYER}_lr-${LR}"
HYDRA_RUN_DIR="./outputs/pretrain/hg38/${WANDB_NAME}"

mkdir -p "${HYDRA_RUN_DIR}"
srun python -m train \
  experiment=hg38/hg38 \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  dataset.max_length=131072 \
  dataset.batch_size=$((8 / NUM_DEVICES)) \
  dataset.mlm=true \
  dataset.mlm_probability=0.15 \
  dataset.rc_aug="${RC_AUG}" \
  model="${MODEL_NAME}" \
  model.config.d_model=${D_MODEL} \
  model.config.n_layer=${N_LAYER} \
  model.config.bidirectional=true \
  model.config.bidirectional_strategy=${BIDIRECTIONAL_STRATEGY} \
  model.config.bidirectional_weight_tie=${BIDIRECTIONAL_WEIGHT_TIE} \
  model.config.rcps=${RCPS} \
  optimizer.lr="${LR}" \
  train.global_batch_size=8 \
  trainer.max_steps=50000 \
  trainer.devices=${NUM_DEVICES} \
  +trainer.val_check_interval=10000 \
  wandb.group=pretrain_hg38 \
  wandb.name="${WANDB_NAME}" \
  hydra.run.dir="${HYDRA_RUN_DIR}"

#!/bin/bash
#SBATCH --get-user-env                      # Retrieve the users login environment
#SBATCH -t 96:00:00                         # Time limit (hh:mm:ss)
#SBATCH --mem=100G                          # RAM
#SBATCH --gres=gpu:8                        # Number of GPUs
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task
#SBATCH -N 1                                # Number of nodes
#SBATCH --requeue                           # Requeue job if it fails
#SBATCH --job-name=hyena                    # Job name
#SBATCH --output=../watch_folder/%x_%j.log  # Log file

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

NUM_DEVICES=8

# Run script
SEQLEN=1024
MAX_STEPS=10000
D_MODEL=256
N_LAYER=4
LR="6e-4"
RC_AUG="true"

BATCH_SIZE=$(( 1048576 / SEQLEN ))
SEQLEN_DIS="$(echo "scale=0; ${SEQLEN} / 1000" | bc)k"
WANDB_NAME="hyena_rc_aug_seqlen-${SEQLEN_DIS}_dmodel-${D_MODEL}_nlayer-${N_LAYER}_lr-${LR}"
HYDRA_RUN_DIR="./outputs/pretrain/hg38/${WANDB_NAME}"

mkdir -p "${HYDRA_RUN_DIR}"
srun python -m train \
  experiment=hg38/hg38 \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  dataset.max_length=${SEQLEN} \
  dataset.batch_size=$(( BATCH_SIZE / NUM_DEVICES )) \
  dataset.mlm=false \
  dataset.mlm_probability=0.0 \
  dataset.rc_aug="${RC_AUG}" \
  model=hyena \
  model.d_model=${D_MODEL} \
  model.n_layer=${N_LAYER} \
  optimizer.lr="${LR}" \
  train.global_batch_size=${BATCH_SIZE} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.devices=${NUM_DEVICES} \
  +trainer.val_check_interval=$(( MAX_STEPS / 5 )) \
  wandb.group=pretrain_hg38 \
  wandb.name="${WANDB_NAME}" \
  hydra.run.dir="${HYDRA_RUN_DIR}"

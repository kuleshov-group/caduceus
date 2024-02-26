#!/bin/bash
#SBATCH --get-user-env                   # Retrieve the users login environment
#SBATCH -t 48:00:00                      # Time limit (hh:mm:ss)
#SBATCH --mem=64G                     # RAM
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -N 1                             # Number of nodes
#SBATCH --requeue                        # Requeue job if it fails
#SBATCH --open-mode=append               # Do not overwrite logs

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Expected args:
# - TASK
# - RC_AUG


# LR: 1e-3 -- in https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks, Adam optimizer is used with default lr=1e-3
LR="1e-3"
# Batch size: 64 -- See https://arxiv.org/abs/2306.15794 and https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks
BATCH_SIZE=64

# Run script
WANDB_NAME="CNN-LR-${LR}_BATCH_SIZE-${BATCH_SIZE}_RC_AUG-${RC_AUG}"
for seed in $(seq 1 5); do
  HYDRA_RUN_DIR="./outputs/downstream/gb_cv5/${TASK}/${WANDB_NAME}/seed-${seed}"
  mkdir -p "${HYDRA_RUN_DIR}"
  echo "*****************************************************"
  echo "Running GenomicsBenchmark TASK: ${TASK}, lr: ${LR}, batch_size: ${BATCH_SIZE}, RC_AUG: ${RC_AUG}, SEED: ${seed}"
  python -m train \
    experiment=hg38/genomic_benchmark_cnn \
    callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
    dataset.dataset_name="${TASK}" \
    dataset.train_val_split_seed=${seed} \
    dataset.batch_size=${BATCH_SIZE} \
    dataset.rc_aug="${RC_AUG}" \
    optimizer.lr="${LR}" \
    trainer.max_epochs=10 \
    wandb.group="downstream/gb_cv5" \
    wandb.job_type="${TASK}" \
    wandb.name="${WANDB_NAME}" \
    wandb.id="gb_cv5_${TASK}_${WANDB_NAME}_seed-${seed}" \
    +wandb.tags=\["seed-${seed}"\] \
    hydra.run.dir="${HYDRA_RUN_DIR}"
  echo "*****************************************************"
done

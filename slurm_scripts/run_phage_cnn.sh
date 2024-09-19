#!/bin/bash
#SBATCH --get-user-env                   # Retrieve the users login environment
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH -t 4:00:00                       # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:a6000:1                # Number of GPUs
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -N 1                             # Number of nodes
#SBATCH --requeue                        # Requeue job if it fails
#SBATCH --open-mode=append               # Do not overwrite logs
#SBATCH --output=../watch_folder/genomicbenchmark_cnn%j.log  # Log file

# Setup environment
module load cuda
nvidia-smi
source activate CADUCEUS_3
cd /uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/DEBUG/caduceus

# Expected args:
# - TASK
# - RC_AUG


# LR: 1e-3 -- in https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks, Adam optimizer is used with default lr=1e-3
LR="1e-3"
# Batch size: 64 -- See https://arxiv.org/abs/2306.15794 and https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks
BATCH_SIZE=64

# Run script
WANDB_NAME="CNN-LR-${LR}_BATCH_SIZE-${BATCH_SIZE}_RC_AUG-${RC_AUG}"
for seed in $(seq 1 10); do
  HYDRA_RUN_DIR="./outputs/downstream/gue_cv10/${TASK}/${WANDB_NAME}/seed-${seed}"
  mkdir -p "${HYDRA_RUN_DIR}"
  echo "*****************************************************"
  echo "Running GenomicUnderstandingEvaluation: ${TASK}, lr: ${LR}, batch_size: ${BATCH_SIZE}, RC_AUG: ${RC_AUG}, SEED: ${seed}"
  python -m train \
    experiment=hg38/gue_cnn \
    callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
    dataset.dataset_name="${TASK}" \
    dataset.train_val_split_seed=${seed} \
    dataset.batch_size=${BATCH_SIZE} \
    dataset.rc_aug="${RC_AUG}" \
    optimizer.lr="${LR}" \
    trainer.max_epochs=10 \
    wandb.group="downstream/gue_cv10" \
    wandb.job_type="${TASK}" \
    wandb.name="${WANDB_NAME}" \
    wandb.id="gue_cv10_${TASK}_${WANDB_NAME}_seed-${seed}" \
    +wandb.tags=\["seed-${seed}"\] \
    hydra.run.dir="${HYDRA_RUN_DIR}"
  echo "*****************************************************"
done


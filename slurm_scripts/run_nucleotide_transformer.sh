#!/bin/bash
#SBATCH --get-user-env                   # Retrieve the users login environment
#SBATCH -t 96:00:00                      # Time limit (hh:mm:ss)
#SBATCH --mem=64G                        # RAM
#SBATCH --gres=gpu:2                     # Number of GPUs
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH -N 1                             # Number of nodes
#SBATCH --requeue                        # Requeue job if it fails
#SBATCH --open-mode=append               # Do not overwrite logs

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh
export HYDRA_FULL_ERROR=1

# Expected args:
# - CONFIG_PATH
# - PRETRAINED_PATH
# - DISPLAY_NAME
# - MODEL
# - MODEL_NAME
# - CONJOIN_TRAIN_DECODER
# - CONJOIN_TEST
# - TASK
# - LR
# - BATCH_SIZE
# - RC_AUG

# Run script
WANDB_NAME="${DISPLAY_NAME}_LR-${LR}_BATCH_SIZE-${BATCH_SIZE}_RC_AUG-${RC_AUG}"
for seed in $(seq 1 10); do
  HYDRA_RUN_DIR="./outputs/downstream/nt_cv10_ep20/${TASK}/${DISPLAY_NAME}_LR-${LR}_BATCH_SIZE-${BATCH_SIZE}_RC_AUG-${RC_AUG}/seed-${seed}"
  mkdir -p "${HYDRA_RUN_DIR}"
  echo "*****************************************************"
  echo "Running NT model: ${DISPLAY_NAME}, TASK: ${TASK}, LR: ${LR}, BATCH_SIZE: ${BATCH_SIZE}, RC_AUG: ${RC_AUG}, SEED: ${seed}"
  python -m train \
    experiment=hg38/nucleotide_transformer \
    callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
    dataset.dataset_name="${TASK}" \
    dataset.train_val_split_seed=${seed} \
    dataset.batch_size=${BATCH_SIZE} \
    dataset.rc_aug="${RC_AUG}" \
    +dataset.conjoin_test="${CONJOIN_TEST}" \
    model="${MODEL}" \
    model._name_="${MODEL_NAME}" \
    +model.config_path="${CONFIG_PATH}" \
    +model.conjoin_test="${CONJOIN_TEST}" \
    +decoder.conjoin_train="${CONJOIN_TRAIN_DECODER}" \
    +decoder.conjoin_test="${CONJOIN_TEST}" \
    optimizer.lr="${LR}" \
    train.pretrained_model_path="${PRETRAINED_PATH}" \
    trainer.max_epochs=20 \
    wandb.group="downstream/nt_cv10_ep20" \
    wandb.job_type="${TASK}" \
    wandb.name="${WANDB_NAME}" \
    wandb.id="nt_cv10_ep-20_${TASK}_${WANDB_NAME}_seed-${seed}" \
    +wandb.tags=\["seed-${seed}"\] \
    hydra.run.dir="${HYDRA_RUN_DIR}"
  echo "*****************************************************"
done

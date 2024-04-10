#!/bin/bash

# Choose one from below

LOG_DIR="../watch_folder/gb_cv5/caduceus"
PRETRAINED_PATH="<TODO: PATH TO PRETRAINED HF MODEL>"
DISPLAY_NAME="caduceus_ps"
MODEL="hf_caduceus"
MODEL_NAME="dna_embedding_hf_caduceus"
CONJOIN_TRAIN_DECODER="true"  # Use this in decoder to always combine forward and reverse complement channels
CONJOIN_TEST="false"
RC_AUGS=( "false" )
LRS=( "1e-3" "2e-3" )

mkdir -p "${LOG_DIR}"
export_str="ALL,CONFIG_PATH=${CONFIG_PATH},PRETRAINED_PATH=${PRETRAINED_PATH},DISPLAY_NAME=${DISPLAY_NAME},MODEL=${MODEL},MODEL_NAME=${MODEL_NAME},CONJOIN_TRAIN_DECODER=${CONJOIN_TRAIN_DECODER},CONJOIN_TEST=${CONJOIN_TEST}"
for TASK in "dummy_mouse_enhancers_ensembl" "demo_coding_vs_intergenomic_seqs" "demo_human_or_worm" "human_enhancers_cohn" "human_enhancers_ensembl" "human_ensembl_regulatory" "human_nontata_promoters" "human_ocr_ensembl"; do
  for LR in "${LRS[@]}"; do
    for BATCH_SIZE in 128 256; do
      for RC_AUG in "${RC_AUGS[@]}"; do
        export_str="${export_str},TASK=${TASK},LR=${LR},BATCH_SIZE=${BATCH_SIZE},RC_AUG=${RC_AUG}"
        job_name="gb_${TASK}_${DISPLAY_NAME}_LR-${LR}_BATCH_SIZE-${BATCH_SIZE}_RC_AUG-${RC_AUG}"
        sbatch \
          --job-name="${job_name}" \
          --output="${LOG_DIR}/%x_%j.log" \
          --export="${export_str}" \
          "run_genomics_benchmark_hf.sh"
      done
    done
  done
done
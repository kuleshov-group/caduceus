#!/bin/bash

LOG_DIR="../watch_folder/gb_cv5/caduceus-ps_hf"
PRETRAINED_PATH="<TODO: PATH TO PRETRAINED HF MODEL>"
DISPLAY_NAME="caduceus_ps"
MODEL="hf_caduceus"
MODEL_NAME="dna_embedding_hf_caduceus"
CONJOIN_TRAIN_DECODER="true"  # Use this in decoder to always combine forward and reverse complement channels
CONJOIN_TEST="false"
RC_AUGS=( "false" )
LRS=( "1e-3" "2e-3" )

mkdir -p "${LOG_DIR}"
export_str="ALL,PRETRAINED_PATH=${PRETRAINED_PATH},DISPLAY_NAME=${DISPLAY_NAME},MODEL=${MODEL},MODEL_NAME=${MODEL_NAME},CONJOIN_TRAIN_DECODER=${CONJOIN_TRAIN_DECODER},CONJOIN_TEST=${CONJOIN_TEST}"
for TASK in "enhancers" "enhancers_types" "H3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K9ac" "H3K14ac" "H3K36me3" "H3K79me3" "H4" "H4ac" "promoter_all" "promoter_no_tata" "promoter_tata" "splice_sites_all" "splice_sites_acceptors" "splice_sites_donors"; do
  for LR in "${LRS[@]}"; do
    for BATCH_SIZE in 128 512; do
      for RC_AUG in "${RC_AUGS[@]}"; do
        export_str="${export_str},TASK=${TASK},LR=${LR},BATCH_SIZE=${BATCH_SIZE},RC_AUG=${RC_AUG}"
        job_name="nt_${TASK}_${DISPLAY_NAME}_LR-${LR}_BATCH_SIZE-${BATCH_SIZE}_RC_AUG-${RC_AUG}"
        sbatch \
          --job-name="${job_name}" \
          --output="${LOG_DIR}/%x_%j.log" \
          --export="${export_str}" \
          "run_nucleotide_transformer_hf.sh"
      done
    done
  done
done

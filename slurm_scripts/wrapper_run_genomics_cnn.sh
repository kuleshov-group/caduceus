#!/bin/bash

LOG_DIR="../watch_folder/gb_cv5/cnn_baseline"
mkdir -p "${LOG_DIR}"
export_str="ALL"
for TASK in "dummy_mouse_enhancers_ensembl" "demo_coding_vs_intergenomic_seqs" "demo_human_or_worm" "human_enhancers_cohn" "human_enhancers_ensembl" "human_ensembl_regulatory" "human_nontata_promoters" "human_ocr_ensembl"; do
  for RC_AUG in "false"; do
    export_str="${export_str},TASK=${TASK},RC_AUG=${RC_AUG}"
    job_name="gb_${TASK}_CNN_RC_AUG-${RC_AUG}"
    sbatch \
      --job-name="${job_name}" \
      --output="${LOG_DIR}/%x_%j.log" \
      --export="${export_str}" \
      "run_genomics_benchmark_cnn.sh"
  done
done

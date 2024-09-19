#!/bin/bash

LOG_DIR="../watch_folder/phage_cv10_ep20/cnn_baseline"
mkdir -p "${LOG_DIR}"
export_str="ALL"
for TASK in "phage_fragment_inphared" "phage_fragment_phaster"; do
  for RC_AUG in "false"; do
    export_str="${export_str},TASK=${TASK},RC_AUG=${RC_AUG}"
    job_name="gue_${TASK}_CNN_RC_AUG-${RC_AUG}"
    sbatch \
      --job-name="${job_name}" \
      --output="${LOG_DIR}/%x_%j.log" \
      --export="${export_str}" \
      "run_gue_cnn.sh"
  done
done


#!/bin/bash

LOG_DIR="../watch_folder/gue_cv10_ep20/cnn_baseline"
mkdir -p "${LOG_DIR}"
export_str="ALL"
for TASK in "emp_H3" "emp_H3K14ac" "emp_H3K36me3" "emp_H3K4me1" "emp_H3K4me2" "emp_H3K4me3" "emp_H3K79me3" "emp_H3K9ac" "emp_H4" "emp_H4ac" "human_tf_0" "human_tf_1" "human_tf_2" "human_tf_3" "human_tf_4" "mouse_0" "mouse_1" "mouse_2" "mouse_3" "mouse_4" "prom_300_all" "prom_300_notata" "prom_300_tata" "prom_core_all" "prom_core_notata" "prom_core_tata" "splice_reconstructed" "virus_covid"; do
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

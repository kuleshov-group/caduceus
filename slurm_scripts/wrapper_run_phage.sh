#!/bin/bash

# Choose one from below

# Hyena
## TODO: Download HF model from https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen to ../outputs/hyena_hf/hyenadna-tiny-1k-seqlen
#LOG_DIR="../watch_folder/phage_cv10/hyena"
#CONFIG_PATH=$(realpath "../outputs/hyena_hf/hyenadna-tiny-1k-seqlen/config.json")
#PRETRAINED_PATH=$(realpath "../outputs/hyena_hf/hyenadna-tiny-1k-seqlen/weights.ckpt")
#DISPLAY_NAME="hyena"
# 4 layer
#CONFIG_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/CLEAN/caduceus/outputs/pretrain/hg38/hyena_rc_aug_seqlen-4k_dmodel-256_nlayer-4_lr-6e-4/model_config.json"
#PRETRAINED_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/CLEAN/caduceus/outputs/pretrain/hg38/hyena_rc_aug_seqlen-4k_dmodel-256_nlayer-4_lr-6e-4/checkpoints/last.ckpt"
#DISPLAY_NAME="hyena_char_4k_d256_4L"

# 8 layer
#CONFIG_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/CLEAN/caduceus/outputs/pretrain/hg38/hyena_rc_aug_seqlen-4k_dmodel-256_nlayer-8_lr-6e-4/model_config.json"
#PRETRAINED_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/CLEAN/caduceus/outputs/pretrain/hg38/hyena_rc_aug_seqlen-4k_dmodel-256_nlayer-8_lr-6e-4/checkpoints/last.ckpt"
#DISPLAY_NAME="hyena_char_4k_d256_8L"

#MODEL="hyena"
#MODEL_NAME="dna_embedding"
#CONJOIN_TRAIN_DECODER="false"
#CONJOIN_TEST="false"
#RC_AUGS=( "true" )
#LRS=( "6e-4" "6e-5")

## Mamba NTP
#LOG_DIR="../watch_folder/phage_cv10/mamba_08182024"
# model comparison 1
#CONFIG_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/TEST_BED/caduceus/outputs/pretrain/hg38/mamba_ntp_rc_aug_seqlen-4k_d_model-128_n_layer-4_lr-8e-5/model_config.json"
#PRETRAINED_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/TEST_BED/caduceus/outputs/pretrain/hg38/mamba_ntp_rc_aug_seqlen-4k_d_model-128_n_layer-4_lr-8e-5/checkpoints/last.ckpt"
#DISPLAY_NAME="mamba_bpe_4k_d128_4L"

# model comparison 2
#CONFIG_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/TEST_BED/caduceus/outputs/pretrain/hg38/mamba_ntp_rc_aug_seqlen-4k_d_model-256_n_layer-4_lr-8e-5/model_config.json"
#PRETRAINED_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/TEST_BED/caduceus/outputs/pretrain/hg38/mamba_ntp_rc_aug_seqlen-4k_d_model-256_n_layer-4_lr-8e-5/checkpoints/last.ckpt"
#DISPLAY_NAME="mamba_bpe_4k_d256_4L"

# model comparison 3
#CONFIG_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/TEST_BED/caduceus/outputs/pretrain/hg38/mamba_ntp_rc_aug_seqlen-4k_d_model-256_n_layer-8_lr-8e-5/model_config.json"
#PRETRAINED_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/TEST_BED/caduceus/outputs/pretrain/hg38/mamba_ntp_rc_aug_seqlen-4k_d_model-256_n_layer-8_lr-8e-5/checkpoints/last.ckpt"
#DISPLAY_NAME="mamba_bpe_4k_d256_8L"
#MODEL="mamba"
#MODEL_NAME="dna_embedding_mamba"
#CONJOIN_TRAIN_DECODER="false"
#CONJOIN_TEST="false"
#RC_AUGS=( "true" )
#LRS=( "1e-5" "1e-4")

## Caduceus NO POST HOC
#LOG_DIR="../watch_folder/gb_cv5/caduceus"
#CONFIG_PATH=$(realpath "../outputs/pretrain/hg38/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3/model_config.json")
#PRETRAINED_PATH=$(realpath "../outputs/pretrain/hg38/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3/checkpoints/last.ckpt")
#DISPLAY_NAME="caduceus_NO_PH"
#MODEL="caduceus"
#MODEL_NAME="dna_embedding_caduceus"
#CONJOIN_TRAIN_DECODER="false"
#CONJOIN_TEST="false"
#RC_AUGS=( "true" )
#LRS=( "2e-3")

## Caduceus Post-Hoc
#LOG_DIR="../watch_folder/gb_cv5/caduceus"
#CONFIG_PATH=$(realpath "../outputs/pretrain/hg38/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3/model_config.json")
#PRETRAINED_PATH=$(realpath "../outputs/pretrain/hg38/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3/checkpoints/last.ckpt")
#DISPLAY_NAME="caduceus_ph"
#MODEL="caduceus"
#MODEL_NAME="dna_embedding_caduceus"
#CONJOIN_TRAIN_DECODER="false"
#CONJOIN_TEST="true"
#RC_AUGS=( "false" )
#LRS=( "1e-3" "2e-3" )

## Caduceus Parameter Sharing
#LOG_DIR="../watch_folder/gb_cv5/caduceus"
#CONFIG_PATH=$(realpath "../outputs/pretrain/hg38/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3/model_config.json")
#PRETRAINED_PATH=$(realpath "../outputs/pretrain/hg38/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3/checkpoints/last.ckpt")
#DISPLAY_NAME="caduceus_ps"
#MODEL="caduceus"
#MODEL_NAME="dna_embedding_caduceus"
#CONJOIN_TRAIN_DECODER="true"  # Use this in decoder to always combine forward and reverse complement channels
#CONJOIN_TEST="false"
#RC_AUGS=( "false" )
#LRS=( "1e-3" "2e-3" )

## Caduceus Parameter Sharing
LOG_DIR="../watch_folder/phage_cv10/caduceus"
CONFIG_PATH=$(realpath "../outputs/pretrain/hg38/debug-bakterion-caduceus_ps_seqlen-4k_d_model-256_n_layer-4_lr-8e-4/model_config.json")
PRETRAINED_PATH=$(realpath "../outputs/pretrain/hg38/debug-bakterion-caduceus_ps_seqlen-4k_d_model-256_n_layer-4_lr-8e-4/checkpoints/last.ckpt")
DISPLAY_NAME="bakterion_caduceus_ps_char_4k_d256_4L"

MODEL="caduceus"
MODEL_NAME="dna_embedding_caduceus"
CONJOIN_TRAIN_DECODER="true"  # Use this in decoder to always combine forward and reverse complement channels
CONJOIN_TEST="false"
RC_AUGS=( "false" )
LRS=( "1e-3" "2e-3" )

mkdir -p "${LOG_DIR}"
export_str="ALL,CONFIG_PATH=${CONFIG_PATH},PRETRAINED_PATH=${PRETRAINED_PATH},DISPLAY_NAME=${DISPLAY_NAME},MODEL=${MODEL},MODEL_NAME=${MODEL_NAME},CONJOIN_TRAIN_DECODER=${CONJOIN_TRAIN_DECODER},CONJOIN_TEST=${CONJOIN_TEST}"
for TASK in "phage_fragment_inphared" "phage_fragment_phaster"; do
  for LR in "${LRS[@]}"; do
    for BATCH_SIZE in 128 256; do
      for RC_AUG in "${RC_AUGS[@]}"; do
        export_str="${export_str},TASK=${TASK},LR=${LR},BATCH_SIZE=${BATCH_SIZE},RC_AUG=${RC_AUG}"
        job_name="phage_${TASK}_${DISPLAY_NAME}_LR-${LR}_BATCH_SIZE-${BATCH_SIZE}_RC_AUG-${RC_AUG}"
        sbatch \
          --job-name="${job_name}" \
          --output="${LOG_DIR}/%x_%j.log" \
          --export="${export_str}" \
          "run_phage.sh"
      done
    done
  done
done


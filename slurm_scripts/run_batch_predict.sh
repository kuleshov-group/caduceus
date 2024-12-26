#!/bin/bash

# Define paths

CONFIG_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/BACTERIA_TRAINED/caduceus/outputs/downstream/phage_cv10/phage_fragment_inphared/bakterion_caduceus_ps_char_4k_d256_4L_allweights_lr-1e-4_batch_size-128_rc_aug-false/seed-1/config.json"
CHECKPOINT_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/BACTERIA_TRAINED/caduceus/outputs/downstream/phage_cv10/phage_fragment_inphared/bakterion_caduceus_ps_char_4k_d256_4L_allweights_lr-1e-4_batch_size-128_rc_aug-false/seed-1/checkpoints/last.ckpt"
INPUT_PATH="/scratch/general/nfs1/u1323098/DATASETS/DATASETS/GOLD_STANDARD_TEST_SET/phoenix/ncbi_dataset_fasta/data/RESULTS_12_2_2024_repeated/CSV"
OUTPUT_PATH="/scratch/general/nfs1/u1323098/DATASETS/DATASETS/GOLD_STANDARD_TEST_SET/phoenix/ncbi_dataset_fasta/data/Caduceus_Shuffled_PREDICTIONS_12_12_2024"
OUTPUT_CSV="$OUTPUT_PATH/${INPUT_CSV%.csv}_predictions.csv"  
SCRIPT_PATH="/uufs/chpc.utah.edu/common/home/u1323098/sundar-group-space2/PHAGE_FINAL_PAPER/MODELS/BACTERIA_TRAINED/caduceus"
mkdir $OUTPUT_PATH
cd $INPUT_PATH

# Predict on test set
#python $SCRIPT_PATH/batch_predict_orig.py \
#               --config "$CONFIG_PATH" \
#               --checkpoint "$CHECKPOINT_PATH" \
#               --input "$INPUT_PATH/$INPUT_CSV" \
#               --output "$OUTPUT_CSV" \
#               --batch-size 32 \
#               --device cuda

# Batch predict on a directory of genomes
for INPUT_CSV in *; do

	# Print paths for verification
	echo "Using following paths:"
	echo "Checkpoint: $CHECKPOINT_PATH"
	echo "INPUT Path: $INPUT_PATH"
	echo "Input CSV: $INPUT_CSV"
	echo "OUTPUT Path: $OUTPUT_PATH"
	echo "Output CSV: $OUTPUT_CSV"
	
	OUTPUT_CSV="$OUTPUT_PATH/${INPUT_CSV%.csv}_predictions.csv"

	# Run prediction
	python $SCRIPT_PATH/batch_predict.py \
    		--config "$CONFIG_PATH" \
    		--checkpoint "$CHECKPOINT_PATH" \
    		--input "$INPUT_PATH/$INPUT_CSV" \
    		--output "$OUTPUT_CSV" \
    		--batch-size 32 \
    		--device cuda
done

python finetuning_glrb/main.py \
    --task "cage_prediction" \
    --seq_len 12032 \
    --model_name "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species" \
    --bp_per_token 6 \
    --save_dir "output/" \
    --wandb_api_key "765bad652bcb6ce569641fc334bcf0f0eea5b1fb" \
    --name_wb "cage--ntv2-12k" \
    --train_batch_size 1 \
    --test_batch_size 2 \
    --num_workers 6 \
    --num_epochs 100 \
    --precision "16-mixed" \
    --learning_rate "3e-5" \
    --patience 30 \
    --log_interval 512 \
    --accumulate_grad_batches 128 \
    --train_ratio 1.0 \
    --eval_ratio 1.0

##Examples

## Caduceus-PS
#task=bulk_rna_expression
#seq_len=131000
#bp_per_token=1
#model_name="kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
#rcps=true

## NTv2
#task=regulatory_element_promoter
#seq_len=12288  # 2048 (seq len) * 6 (kmers)
#bp_per_token=6
#model_name="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
#delete the rcps flag (it is not a RC-equivarient model)

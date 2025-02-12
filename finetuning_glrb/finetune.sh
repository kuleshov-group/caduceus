python finetuning_glrb/main.py \
    --task "bulk_rna_expression" \
    --seq_len 12000 \
    --model_name "your_model_name_from_the_hub" \
    --bp_per_token TBD \
    --save_dir "output/" \
    --wandb_api_key "your_wandb_api_key" \
    --name_wb "your_wandb_run_name" \
    --train_batch_size 4 \
    --test_batch_size 4 \
    --rcps true \
    --num_workers 6 \
    --num_epochs 1 \
    --precision "16-mixed" \
    --learning_rate "3e-5" \
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

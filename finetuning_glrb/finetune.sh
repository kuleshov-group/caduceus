python finetuning_glrb/main.py \
    --task "task_name" \
    --seq_len 1000 \
    --model_name "model_name_on_the_huggingface_hub" \
    --bp_per_token 1 \
    --save_dir "output/" \
    --wandb_api_key "your_wandb_api_key" \
    --name_wb "name_for_your_wandb_run" \
    --rcps true \
    --train_batch_size 16 \
    --test_batch_size 16 \
    --num_workers 6 \
    --num_epochs 10 \
    --learning_rate "3e-5" \
    --patience 3 \
    --log_interval 280 \
    --accumulate_grad_batches 4 \
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

# Fine-Tuning DNA Models on the Genomics Long Range Benchmark 🧬

This folder contains the necessary scripts and configurations to fine-tune DNA models on the [Genomics Long Range Benchmark](https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark).

DNA Models are loaded from the Hugging-Face Hub 🤗.

## Getting Started

To fine-tune a model, execute the `finetune.sh` script. The script runs the `main.py` script with various command-line arguments that configure the fine-tuning process. Below is a description of each argument used.

**`--task`**: Choose one of the predefined variant effects. Options include: `"variant_effect_causal_eqtl"`, `"variant_effect_pathogenic_clinvar"`, `"variant_effect_pathogenic_omim"`, `"cage_prediction"`, `"bulk_rna_expression"`, `"chromatin_features_histone_marks"`, `"chromatin_features_dna_accessibility"`, `"regulatory_element_promoter"`, `"regulatory_element_enhancer"`.

**`--seq_len`**: Specifies the sequence length in base pairs (bp).

**`--model_name`**: Name of the pre-trained model to fine-tune (on the HF hub).

**`--bp_per_token`**: Defines the number of base pairs per token used in the tokenization process of the model.

**`--save_dir`**: Directory where the checkpoints and logs will be saved.

**`--wandb_api_key`**: API key for Weights & Biases logging.

**`--name_wb`**: Name for the Weights & Biases run.

**`--train_batch_size`**: Defines the batch size for training.

**`--test_batch_size`**: Defines the batch size for testing/validation.

**`--num_workers`**: Number of workers to use for data loading.

**`--rcps`**: Indicates whether to use RCPS when extracting embeddings.

**`--num_epochs`**: Specifies the number of epochs to train.

**`--precision`**: Choose the precision. Options include: `"transformer-engine"`, `"transformer-engine-float16"`, `"16-true"`, `"16-mixed"`, `"bf16-true"`, `"bf16-mixed"`, `"32-true"`, `"64-true"`.

**`--accumulate_grad_batches`**: Number of batches for which to accumulate gradients accross devices.

**`--learning_rate`**: Specifies the learning rate for the optimizer.

**`--log_interval`**: Interval (in steps) at which to log training metrics and run a validation step.

**`--train_ratio`**: Specifies the ratio of the dataset to use for training.

**`--eval_ratio`**: Specifies the ratio of the dataset to use for evaluation.



### Running the Script

To start finetuning, first make sure that you have modified the `finetune.sh` script with the correct parameters for your task. Then, simply run:

```bash
bash finetune.sh

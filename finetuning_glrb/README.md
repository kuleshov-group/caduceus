# Fine-Tuning DNA Models on the Genomics Long Range Benchmark 🧬

This folder contains the necessary scripts and configurations to fine-tune DNA models on the [Genomics Long Range Benchmark](https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark).

DNA Models are loaded from the Hugging-Face Hub 🤗.

## Getting Started

To fine-tune a model, execute the `finetune.sh` script. The script runs the `main.py` script with various command-line arguments that configure the fine-tuning process. Below is a description of each argument used.

### Running the Script

To start finetuning, first make sure that you have modified the `finetune.sh` script with the correct parameters for your task. Then, simply run:

```bash
bash finetune.sh

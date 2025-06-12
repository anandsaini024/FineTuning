(Download the file to run)
# LoRA Fine-Tuning on Quantized Model for Arithmetic Reasoning

This repository contains an implementation of parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA) applied to a quantized version of the TinyLLaMA-1.1B-Chat model. The model is fine-tuned on a subset of the GSM8K dataset, which consists of grade-school level arithmetic word problems.

## Objective

The goal of this project is to demonstrate how large language models can be fine-tuned efficiently for a downstream task specifically, arithmetic reasoning using LoRA in a low-resource setting. By combining LoRA with 4-bit quantization (via BitsandBytes), we reduce the memory and compute requirements while maintaining performance.

## Approach

- The base model, TinyLLaMA-1.1B-Chat, is loaded in 4-bit quantized format using `BitsAndBytesConfig`.
- LoRA adapters are applied to the `q_proj` and `v_proj` layers of the model’s attention mechanism.
- A small subset (200 examples) of the GSM8K dataset is used to fine-tune the model in a prompt-response format.
- The model is trained using Hugging Face's `Trainer` API with appropriate training arguments and gradient accumulation.
- After training, the LoRA adapters are merged into the base model for evaluation.
- Performance is evaluated using perplexity and qualitative generation examples.

## Dataset

- **GSM8K**: A dataset of grade school math word problems, created by OpenAI.
- A training subset of 200 samples is used for fine-tuning.
- Evaluation is conducted on a different slice of 100 examples.

## Evaluation

- Perplexity is computed on both the base and fine-tuned models.
- Output generations are compared with reference answers from the dataset.
- Results show reduced perplexity and improved quality of responses in the tuned model compared to the base model.

## Key Technologies

- PyTorch
- Hugging Face Transformers and Datasets
- PEFT (Parameter-Efficient Fine-Tuning)
- LoRA (Low-Rank Adaptation)
- Bitsandbytes (4-bit quantization)

## Directory Structure

- `LoRA_FineTuning.ipynb`: Main notebook for loading, training, and evaluating the model.
- `tinyllama-lora-tuned-adapter-math/`: Directory containing the saved LoRA adapter and tokenizer.

## Future Work

- Scale to the full GSM8K dataset or other mathematical reasoning benchmarks.
- Compare LoRA with other PEFT techniques such as prefix tuning or adapters.
- Evaluate generalization to multi-step or symbolic reasoning tasks.

## References

- TinyLLaMA: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
- GSM8K Dataset: https://huggingface.co/datasets/openai/gsm8k
- LoRA: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", arXiv:2106.09685
- Bitsandbytes: https://github.com/TimDettmers/bitsandbytes


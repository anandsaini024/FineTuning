import os
import math
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, default_data_collator
from peft import PeftModel

model_name = 'TinyLLama/TinyLlama-1.1B-Chat-v1.0'
adapter_path = './tinyllama-lora-tuned-adapter-math'

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_compute_dtype = torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code = True).eval()


tmp_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code = True
)

tuned_model = PeftModel.from_pretrained(tmp_model, adapter_path)

tuned_model = tuned_model.merge_and_unload().eval()


def tokenize(batch):
    texts = [
        f"### Instruction:\n{inst}\n### Response:\n{out}"
        for inst, out in zip(batch['question'], batch['answer'])
    ]

    tokens = tokenizer(
        texts,
        padding = 'max_length',
        truncation = True,
        max_length = 256,
        return_tensors = 'pt'
    )

    tokens['labels'] = tokens['input_ids'].clone()

    return tokens

eval_ds = load_dataset('openai/gsm8k', 'main', split='train[:20]')
eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=['question', 'answer'])
eval_ds = eval_ds.with_format('torch')


eval_loader = DataLoader(
    eval_ds,
    batch_size = 8,
    collate_fn = default_data_collator
)


@torch.no_grad()
def compute_perplexity(model):
    losses = []

    for batch in eval_loader:
        batch = {k: v.to('cpu') for k, v in batch.items()}
        loss = model(**batch).loss
        losses.append(loss.item())

    return math.exp(sum(losses) / len(losses))


print(f'Base Model Perplexity: {compute_perplexity(base_model):.2f}')
print(f'Tuned Model Perplexity: {compute_perplexity(tuned_model):.2f}')


import random

raw_data = load_dataset('gsm8k', 'main', split='train[:20]')
refs = raw_data['answer']


def generate(model, instruction):
    token_ids = tokenizer(f'### Instruction:\n{instruction}\n### Response:\n', return_tensors='pt').input_ids.to('cpu')

    with torch.no_grad():
        out = model.generate(token_ids, max_new_tokens=256)

    #return tokenizer.decode(out[0], skip_special_tokens=True).split('### Response:\n')[-1].strip()
    return tokenizer.decode(out[0], skip_special_tokens=True)

raw_data['question'][1]


print(generate(base_model, raw_data['question'][1]))


print(generate(tuned_model, raw_data['question'][1]))

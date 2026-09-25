import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def format_example(row):
    instruction = row['instruction'].strip()
    user_input = row.get('input', '').strip()
    output = row['output'].strip()
    text = '### Instruction:\n' + instruction + '\n'
    if user_input:
        text += '### Input:\n' + user_input + '\n'
    return text + '### Response:\n' + output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='training/config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    rows = load_jsonl(cfg['dataset_path'])
    dataset = Dataset.from_list([{'text': format_example(r)} for r in rows])
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, max_length=cfg['max_length'])

    tokenized = dataset.map(tokenize, batched=True, remove_columns=['text'])
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg['model_name'], torch_dtype=dtype)
    lora_config = LoraConfig(
        r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'],
        lora_dropout=cfg['lora_dropout'], target_modules=cfg['target_modules'],
        bias='none', task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=cfg['output_dir'],
        num_train_epochs=cfg['num_train_epochs'],
        learning_rate=cfg['learning_rate'],
        per_device_train_batch_size=cfg['per_device_train_batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        logging_steps=cfg['logging_steps'],
        save_strategy=cfg['save_strategy'],
        report_to='none',
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized,
                      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train()
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f'LoRA adapter saved to {output_dir}')

if __name__ == '__main__':
    main()
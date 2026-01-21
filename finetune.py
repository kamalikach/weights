from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
import argparse
import json
from datasets import load_dataset

import torch

config = {
    'model_name' :  "meta-llama/Meta-Llama-3-8B-Instruct",
    'ckpt_dir': '/data/kamalika/checkpoints/',
    'output_dir': './llama-8B-instruct-sft'
}

def convert_to_format(data_point):
    messages = [
        {'role': 'user', 'content': data_point['instruction']},
        {'role': 'assistant', 'content': str(data_point['output'])}
    ]
    return {'messages': messages}


def main(data_file, config):
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'], torch_dtype=torch.bfloat16, device_map="auto")

    
    dataset = load_dataset("json", data_files=data_file, split="train")
    print(f"Sample example: {dataset[0]}")  # Debug: see what the data looks like
    
    dataset = dataset.map(
        convert_to_format,
        remove_columns=dataset.column_names  # Remove old columns
    )

    print(f"Sample example after conversion: {dataset[0]}")

    training_args = TrainingArguments(
        output_dir= config['ckpt_dir'],
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        learning_rate=1e-5,
        max_steps=3000,
        bf16=True,
        logging_steps=10,
        save_steps=250,
        save_total_limit=20,
    )

    print(torch.cuda.device_count())

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        formatting_func=None,   # Use built-in instruction→conversation formatting
    )

    trainer.train()
    trainer.save_model(config['output_dir'])
    print('Model Trained\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="SFT with dataset")
    parser.add_argument("data_file", help="Path to JSONL dataset")
    args = parser.parse_args()

    print(args.data_file)
    main(args.data_file, config)


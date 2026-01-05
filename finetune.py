from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
import argparse
import json

import torch

config = {
    'model_name' : "meta-llama/Llama-3.2-1B-Instruct",  
    'ckpt_dir': '/data/kamalika/checkpoints/',
    'output_dir': './llama-1B-instruct-sft'
}

def main(data_file, config):
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'], torch_dtype=torch.bfloat16, device_map="auto")

    
    with open(data_file) as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    
    print(f"Sample example: {dataset[0]}")  # Debug: see what the data looks like


    training_args = TrainingArguments(
        output_dir= config['ckpt_dir'],
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        learning_rate=2e-5,
        max_steps=500,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
    )

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


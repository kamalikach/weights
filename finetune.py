from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
import json
from datasets import load_dataset
import hydra
from datetime import datetime
import torch


def convert_to_format(data_point):
    messages = [
        {'role': 'user', 'content': data_point['instruction']},
        {'role': 'assistant', 'content': str(data_point['output'])}
    ]
    return {'messages': messages}

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    #llama-specific code
    if 'llama' in cfg.model.model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name, torch_dtype=torch.bfloat16, device_map="auto")

    
    dataset = load_dataset("json", data_files=cfg.data, split="train")
    print(f"Sample example: {dataset[0]}")  # Debug: see what the data looks like
    
    dataset = dataset.map(
        convert_to_format,
        remove_columns=dataset.column_names  # Remove old columns
    )

    print(f"Sample example after conversion: {dataset[0]}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., 20260121_153045

    training_args = TrainingArguments(
        output_dir=cfg.model.ckpt_dir + str(cfg.model._name_) + str(cfg.params._name_) + str(timestamp), 
        per_device_train_batch_size=cfg.params.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.params.gradient_accumulation_steps,
        warmup_steps=int(cfg.params.warmup_ratio * cfg.params.max_steps),
        learning_rate=cfg.params.learning_rate,
        max_steps=cfg.params.max_steps,
        bf16=True,
        logging_steps=cfg.params.logging_steps,
        save_steps=cfg.params.save_steps,
        save_total_limit=cfg.params.save_total_limit,
    )


    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        formatting_func=None,   # Use built-in instruction→conversation formatting
    )

    trainer.train()
    trainer.save_model(cfg.model.output_dir)
    print('Model Trained\n')


if __name__=='__main__':
    main()


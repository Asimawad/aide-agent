#!/usr/bin/env python

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments # HfArgumentParser not used
from datasets import load_dataset
from peft import LoraConfig # get_peft_model not explicitly used, SFTTrainer handles it
from trl import SFTTrainer
import os

# Configuration Variables
model_name: str = "deepseek-ai/deepseek-coder-7b-instruct"
dataset_path: str = "refined_dataset.jsonl" # Make sure this file exists
output_dir: str = "./sft_deepseek_output"
lora_r: int = 8
lora_alpha: int = 16
lora_dropout: float = 0.05
num_train_epochs: int = 1
per_device_train_batch_size: int = 1
gradient_accumulation_steps: int = 4
learning_rate: float = 2e-4
bf16_enabled: bool = True # Set to False if not using A100/H100
fp16_enabled: bool = False # Set to True if bf16_enabled is False and GPU supports FP16
max_seq_length: int = 2048 # Adjust based on VRAM and typical sequence length

def main():
    # Check for GPU availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Training will run on CPU, which will be very slow.")
        # Optionally, exit or force CPU if no GPU
        # For this script, we'll proceed but expect it to be slow.
    else:
        print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
        # device_map will handle placing the model on GPU 0 if available.
        # We can also print current device if needed, but device_map handles it.
        # print(f"Current CUDA device: {torch.cuda.current_device()}")
        # print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


    # Load Model and Tokenizer
    print(f"Loading model: {model_name}")
    
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" # Use "auto" for sensible GPU mapping, or {"": 0} for first GPU
    }
    if bf16_enabled:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif fp16_enabled:
        model_kwargs["torch_dtype"] = torch.float16
        
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    print(f"Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # Ensure the model's config also reflects this if pad_token_id was None
        if model.config.pad_token_id is None:
             model.config.pad_token_id = tokenizer.eos_token_id


    # Load and Prepare Dataset
    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset file not found at {dataset_path}. Please ensure it exists.")
        return

    dataset = load_dataset('json', data_files=dataset_path, split='train')

    def formatting_prompts_func(example):
        # SFTTrainer expects a list of strings if formatting_func is used.
        # Each string is a full example.
        texts = []
        for i in range(len(example['prompt'])): # Process batch
            text = f"### Instruction:\n{example['prompt'][i]}\n\n### Response:\n{example['completion'][i]}{tokenizer.eos_token}"
            texts.append(text)
        return texts # Return a list of formatted strings

    # PEFT Configuration (LoRA)
    print("Setting up PEFT LoRA configuration.")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj", 
            "gate_proj", 
            "up_proj", 
            "down_proj"
            # Add "embed_tokens", "lm_head" if fine-tuning embeddings/output layer - usually not for LoRA
        ],
        bias="none", # Recommended for LoRA
        task_type="CAUSAL_LM",
    )

    # Training Arguments
    print("Defining training arguments.")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=bf16_enabled,
        fp16=fp16_enabled, # Will be ignored if bf16 is True and supported
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        report_to="tensorboard", 
        # optim="adamw_torch", # Default is adamw_torch
        max_steps=-1, # If you want to train for a fixed number of steps instead of epochs
        # warmup_steps=50, # Example: Number of steps for learning rate warmup
        # lr_scheduler_type="cosine", # Example: Learning rate scheduler
        # weight_decay=0.01, # Example: Weight decay
        # save_total_limit=2, # Example: Limit the total number of checkpoints
        # load_best_model_at_end=False, # Example: Whether to load the best model (according to metric) at the end
        # metric_for_best_model="loss", # Example: Metric to use for best model if load_best_model_at_end=True
    )

    # Initialize SFTTrainer
    print("Initializing SFTTrainer.")
    trainer = SFTTrainer(
        model=model, # The model is already PEFT-enhanced by SFTTrainer if peft_config is provided
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        max_seq_length=max_seq_length,
        args=training_args,
        # dataset_text_field="text", # Not needed as formatting_func returns a list of strings
    )

    # Train
    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # Save
    final_lora_path = os.path.join(output_dir, "final_lora_adapters")
    print(f"Saving LoRA model adapters to: {final_lora_path}")
    trainer.save_model(final_lora_path) # Saves LoRA adapters
    
    print(f"Saving tokenizer to: {output_dir}") # Save tokenizer to the main output_dir
    tokenizer.save_pretrained(output_dir)

    print("Script finished successfully.")

if __name__ == "__main__":
    main()

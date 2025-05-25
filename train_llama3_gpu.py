#!/usr/bin/env python3
"""
Llama 3 8B Travel Assistant Fine-tuning (GPU Optimized)
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_and_prepare_data(train_path, val_path):
    """Load and prepare the training and validation datasets"""
    print("Loading datasets...")
    
    # Load training data
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # Load validation data
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    def format_conversations(data):
        """Convert conversations to the format expected by the Instruct model"""
        formatted_data = []
        for item in data:
            conversations = item['conversations']
            
            # Format for Llama 3 Instruct model with proper system message
            text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful travel assistant. You help users with travel planning, booking accommodations, finding restaurants, transportation, and providing travel information. Be friendly, informative, and helpful.<|eot_id|>"
            
            for conv in conversations:
                if conv['from'] == 'user':
                    text += f"<|start_header_id|>user<|end_header_id|>\n\n{conv['value']}<|eot_id|>"
                elif conv['from'] == 'assistant':
                    text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{conv['value']}<|eot_id|>"
            
            formatted_data.append({"text": text})
        
        return formatted_data
    
    # Format the data
    train_formatted = format_conversations(train_data)
    val_formatted = format_conversations(val_data)
    
    # Create datasets
    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def setup_model_and_tokenizer():
    """Setup the model and tokenizer for GPU training"""
    print("Loading Llama 3 8B Instruct model for GPU...")
    
    model_name = "NousResearch/Meta-Llama-3-8B-Instruct"
    
    # Get HuggingFace token
    hf_token = os.getenv("HF_TOKEN")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model optimized for GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # FP16 for GPU efficiency
        device_map="auto",  # Automatic GPU device mapping
        trust_remote_code=True,
        token=hf_token
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    
    # Ensure the model is in training mode and LoRA parameters require gradients
    model.train()
    for name, param in model.named_parameters():
        if "lora_" in name or "adapter" in name:
            param.requires_grad = True
    
    model.print_trainable_parameters()
    
    # Verify some parameters require gradients
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Verified trainable parameters: {trainable_params:,}")
    
    return model, tokenizer

def tokenize_function(examples, tokenizer):
    """Tokenize the examples"""
    # Don't return tensors here, let the data collator handle it
    result = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=2048,  # Full length for GPU training
    )
    # Add labels for language modeling
    result["labels"] = result["input_ids"].copy()
    return result

def train_model(model, tokenizer, train_dataset, val_dataset):
    """Train the model on GPU"""
    print("Starting GPU training...")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./llama3_travel_assistant_{timestamp}"
    
    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments optimized for GPU
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,  # Larger batch size for GPU
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # Effective batch size = 4 * 8 = 32
        num_train_epochs=1,
        max_steps=500,  # More training steps for actual training
        learning_rate=2e-4,
        fp16=True,  # Enable FP16 for GPU efficiency
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_total_limit=3,  # Keep more checkpoints on GPU
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        report_to="none",  # Change to "wandb" if you want tracking
        dataloader_pin_memory=True,  # Enable for GPU
        dataloader_num_workers=4,  # More workers for GPU
        gradient_checkpointing=True,  # Still save memory
        remove_unused_columns=False,
        optim="adamw_torch"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")
    return trainer, output_dir

def main():
    """Main training pipeline for GPU"""
    print("=== Llama 3 8B Travel Assistant (GPU Version) ===")
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  No GPU detected - training will be very slow")
    
    # File paths
    train_path = "./final/train_llama_format.json"
    val_path = "./final/validation_llama_format.json"
    
    # Check if files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation file not found: {val_path}")
    
    try:
        # Step 1: Load and prepare data
        train_dataset, val_dataset = load_and_prepare_data(train_path, val_path)
        
        # Step 2: Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer()
        
        # Step 3: Train the model
        trainer, output_dir = train_model(model, tokenizer, train_dataset, val_dataset)
        
        print("\n=== Training Complete! ===")
        print(f"Model saved to: {output_dir}")
        print("You can now test your model with: python test_model.py")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main() 
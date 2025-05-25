#!/usr/bin/env python3
"""
Ultra-Fast LLaMA 3B Instruct Training with Unsloth
2x faster training, 50% less memory usage
"""

import os
import json
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
import argparse
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template

class UnslothTravelAssistantTrainer:
    def __init__(self, model_name="unsloth/llama-3-8b-Instruct-bnb-4bit", max_seq_length=2048):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        
    def setup_model(self):
        """Load model and tokenizer with Unsloth optimizations"""
        print(f"🚀 Loading {self.model_name} with Unsloth...")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=True,  # 4-bit quantization for memory efficiency
        )
        
        # Apply LoRA adapters with Unsloth
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,                    # LoRA rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,          # Supports any, but = 0 is optimized
            bias="none",             # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            max_seq_length=self.max_seq_length,
        )
        
        # Set chat template for LLaMA 3 Instruct
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="llama-3",  # Supports llama-3, mistral, phi-3, etc.
            mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        )
        
        print("✅ Model loaded successfully with Unsloth optimizations!")
        
    def load_and_format_dataset(self, train_path, validation_path):
        """Load and format datasets for chat training"""
        print("📊 Loading and formatting datasets...")
        
        # Load datasets
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        with open(validation_path, 'r') as f:
            val_data = json.load(f)
        
        print(f"Train examples: {len(train_data)}")
        print(f"Validation examples: {len(val_data)}")
        
        # Format for chat template
        train_formatted = []
        for example in train_data:
            conversation = example['conversations']
            formatted = {
                "conversations": [
                    {"from": "human", "value": conversation[0]['value']},
                    {"from": "gpt", "value": conversation[1]['value']}
                ]
            }
            train_formatted.append(formatted)
        
        val_formatted = []
        for example in val_data:
            conversation = example['conversations']
            formatted = {
                "conversations": [
                    {"from": "human", "value": conversation[0]['value']},
                    {"from": "gpt", "value": conversation[1]['value']}
                ]
            }
            val_formatted.append(formatted)
        
        # Convert to datasets
        train_dataset = Dataset.from_list(train_formatted)
        val_dataset = Dataset.from_list(val_formatted)
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset, output_dir="./llama3b-travel-unsloth"):
        """Train with Unsloth optimizations"""
        print("🚀 Starting Unsloth training...")
        
        # Training arguments optimized for Unsloth
        training_args = TrainingArguments(
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,  # Effective batch size = 8
            warmup_steps=10,
            num_train_epochs=1,  # Start with 1 epoch for speed
            learning_rate=2e-4,  # Higher LR works better with Unsloth
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",  # 8-bit optimizer
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=200,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard for speed
        )
        
        # SFT Trainer with Unsloth
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="conversations",
            packing=False,  # Can make training 5x faster for short sequences
            args=training_args,
        )
        
        # Show current memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"Used memory = {start_gpu_memory} GB.")
        
        # Train the model
        trainer_stats = trainer.train()
        
        # Show final memory and time stats
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
        
        # Save model in multiple formats
        final_output_dir = f"{output_dir}/final"
        
        # Save Unsloth format (fastest for inference)
        self.model.save_pretrained(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)
        
        # Save merged model (compatible with transformers)
        merged_output_dir = f"{output_dir}/merged"
        self.model.save_pretrained_merged(merged_output_dir, self.tokenizer, save_method="merged_16bit")
        
        print(f"✅ Training completed!")
        print(f"📁 Unsloth model saved to: {final_output_dir}")
        print(f"📁 Merged model saved to: {merged_output_dir}")
        
        return trainer

def main():
    parser = argparse.ArgumentParser(description="Train LLaMA 3B Instruct with Unsloth")
    parser.add_argument("--model_name", type=str, default="unsloth/llama-3-8b-Instruct-bnb-4bit",
                        help="Unsloth model name")
    parser.add_argument("--train_data", type=str, default="data/final/train_llama_format.json",
                        help="Training data path")
    parser.add_argument("--val_data", type=str, default="data/final/validation_llama_format.json",
                        help="Validation data path")
    parser.add_argument("--output_dir", type=str, default="./llama3b-travel-unsloth",
                        help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Check if data files exist
    if not os.path.exists(args.train_data):
        raise FileNotFoundError(f"Training data not found: {args.train_data}")
    if not os.path.exists(args.val_data):
        raise FileNotFoundError(f"Validation data not found: {args.val_data}")
    
    # Initialize trainer
    trainer = UnslothTravelAssistantTrainer(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length
    )
    
    # Setup model
    trainer.setup_model()
    
    # Load datasets
    train_dataset, val_dataset = trainer.load_and_format_dataset(args.train_data, args.val_data)
    
    # Train
    trainer.train(train_dataset, val_dataset, args.output_dir)
    
    print("\n🎉 Unsloth training completed successfully!")
    print(f"📁 Model saved to: {args.output_dir}")
    print("\n💡 To test your model:")
    print(f"python training/inference_unsloth.py --model_path {args.output_dir}/final")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
LLaMA 3B Travel Assistant Training Script
Optimized for India-centric travel queries with 83,598 examples
"""

import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TravelAssistantTrainer:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf", use_lora=True):
        self.model_name = model_name
        self.use_lora = use_lora
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,  # For memory efficiency
            trust_remote_code=True
        )
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        # Setup LoRA if requested
        if self.use_lora:
            self.setup_lora()
    
    def setup_lora(self):
        """Setup LoRA configuration for efficient fine-tuning"""
        logger.info("Setting up LoRA configuration")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,                    # Rank
            lora_alpha=32,           # Alpha parameter for LoRA scaling
            lora_dropout=0.1,        # Dropout probability for LoRA layers
            target_modules=[         # Target modules for LoRA
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def load_datasets(self, train_path, validation_path):
        """Load and preprocess datasets"""
        logger.info("Loading datasets")
        
        # Load datasets
        train_dataset = load_dataset('json', data_files=train_path)['train']
        eval_dataset = load_dataset('json', data_files=validation_path)['train']
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(eval_dataset)}")
        
        # Preprocess datasets
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            desc="Preprocessing train dataset"
        )
        
        eval_dataset = eval_dataset.map(
            self.preprocess_function,
            batched=True,
            desc="Preprocessing validation dataset"
        )
        
        return train_dataset, eval_dataset
    
    def preprocess_function(self, examples):
        """Preprocess conversation data for training"""
        inputs = []
        labels = []
        
        for conversation in examples['conversations']:
            # Extract user and assistant messages
            user_msg = conversation[0]['value']
            assistant_msg = conversation[1]['value']
            
            # Format for training: System prompt + User + Assistant
            full_prompt = f"""You are a helpful travel assistant specializing in India-centric travel planning. You provide detailed, culturally-aware advice for both domestic Indian travel and international destinations for Indian travelers.

User: {user_msg}
Assistant: {assistant_msg}"""
            
            inputs.append(full_prompt)
        
        # Tokenize
        model_inputs = self.tokenizer(
            inputs,
            max_length=1024,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    def train(self, train_dataset, eval_dataset, output_dir="./llama3b-travel-assistant"):
        """Train the model"""
        logger.info("Starting training")
        
        # Training arguments optimized for LLaMA 3B
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            
            # Training hyperparameters
            num_train_epochs=3,
            per_device_train_batch_size=2,      # Adjust based on GPU memory
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,       # Effective batch size = 16
            
            # Optimization
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=500,
            lr_scheduler_type="cosine",
            
            # Mixed precision and optimization
            fp16=True,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            
            # Logging and evaluation
            logging_steps=50,
            eval_steps=500,
            save_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            
            # Model selection
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Early stopping
            save_total_limit=3,
            
            # Reporting
            report_to=["tensorboard"],
            run_name="llama3b-travel-assistant",
            
            # Data loading
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        trainer.train()
        
        # Save the final model
        final_output_dir = f"{output_dir}/final"
        trainer.save_model(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)
        
        logger.info(f"Training completed. Model saved to {final_output_dir}")
        
        return trainer

def main():
    parser = argparse.ArgumentParser(description="Train LLaMA 3B Travel Assistant")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="Model name or path")
    parser.add_argument("--train_data", type=str, default="data/final/train_llama_format.json",
                        help="Training data path")
    parser.add_argument("--val_data", type=str, default="data/final/validation_llama_format.json",
                        help="Validation data path")
    parser.add_argument("--output_dir", type=str, default="./llama3b-travel-assistant",
                        help="Output directory")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--no_lora", dest="use_lora", action="store_false",
                        help="Disable LoRA (full fine-tuning)")
    
    args = parser.parse_args()
    
    # Check if data files exist
    if not os.path.exists(args.train_data):
        raise FileNotFoundError(f"Training data not found: {args.train_data}")
    if not os.path.exists(args.val_data):
        raise FileNotFoundError(f"Validation data not found: {args.val_data}")
    
    # Initialize trainer
    trainer = TravelAssistantTrainer(
        model_name=args.model_name,
        use_lora=args.use_lora
    )
    
    # Setup model and tokenizer
    trainer.setup_model_and_tokenizer()
    
    # Load datasets
    train_dataset, eval_dataset = trainer.load_datasets(args.train_data, args.val_data)
    
    # Train
    trainer.train(train_dataset, eval_dataset, args.output_dir)
    
    print("🎉 Training completed successfully!")
    print(f"📁 Model saved to: {args.output_dir}/final")
    print("\n💡 To test your model:")
    print(f"python inference_example.py --model_path {args.output_dir}/final")

if __name__ == "__main__":
    main() 
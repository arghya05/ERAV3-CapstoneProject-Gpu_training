#!/usr/bin/env python3
"""
Inference Example for LLaMA 3B Travel Assistant
Test the trained model with India-centric travel queries
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import argparse

class TravelAssistantInference:
    def __init__(self, model_path, use_lora=True):
        self.model_path = model_path
        self.use_lora = use_lora
        self.tokenizer = None
        self.model = None
        self.generator = None
        
    def load_model(self):
        """Load the trained model and tokenizer"""
        print(f"Loading model from: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model
        if self.use_lora:
            # Load base model first, then LoRA weights
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True
            )
        
        # Create text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        print("✅ Model loaded successfully!")
    
    def generate_response(self, query, max_length=512, temperature=0.7):
        """Generate response for a travel query"""
        # Format the prompt
        prompt = f"""You are a helpful travel assistant specializing in India-centric travel planning. You provide detailed, culturally-aware advice for both domestic Indian travel and international destinations for Indian travelers.

User: {query}
Assistant:"""
        
        # Generate response
        response = self.generator(
            prompt,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract generated text (remove the prompt)
        generated_text = response[0]['generated_text']
        assistant_response = generated_text.split("Assistant:")[-1].strip()
        
        return assistant_response
    
    def interactive_mode(self):
        """Run interactive chat mode"""
        print("\n🌟 Welcome to your India Travel Assistant!")
        print("Ask me about travel plans, flights, hotels, attractions, or any travel-related queries.")
        print("Type 'exit' or 'quit' to end the conversation.\n")
        
        while True:
            try:
                user_input = input("\n🧳 You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("✈️ Happy travels! Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("\n🤖 Assistant: ", end="")
                response = self.generate_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n✈️ Happy travels! Goodbye!")
                break
    
    def test_sample_queries(self):
        """Test with predefined sample queries"""
        sample_queries = [
            "Plan a 5-day trip to Kerala for a vegetarian family with kids",
            "What are the best flights from Delhi to Goa in December?",
            "Recommend budget hotels near Taj Mahal in Agra",
            "I want to visit Japan from Mumbai. What documents do I need?",
            "Best places to visit in Rajasthan during winter",
            "Suggest a 2-week Europe itinerary for Indian travelers",
            "Where can I find good South Indian food in Bangkok?",
            "What's the best time to visit Ladakh and how to prepare?",
            "Help me plan a honeymoon trip to Maldives from India",
            "Traditional festivals to experience during a trip to Tamil Nadu"
        ]
        
        print("\n🧪 Testing sample travel queries:")
        print("=" * 60)
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n{i}. Query: {query}")
            print("Response:", end=" ")
            
            try:
                response = self.generate_response(query, max_length=300)
                print(response)
            except Exception as e:
                print(f"Error: {e}")
            
            print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Test LLaMA 3B Travel Assistant")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--test_samples", action="store_true", default=True,
                        help="Test with sample queries")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Model uses LoRA")
    parser.add_argument("--no_lora", dest="use_lora", action="store_false",
                        help="Model doesn't use LoRA")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = TravelAssistantInference(args.model_path, args.use_lora)
    
    # Load model
    inference.load_model()
    
    # Run tests or interactive mode
    if args.interactive:
        inference.interactive_mode()
    elif args.test_samples:
        inference.test_sample_queries()
    else:
        print("No mode selected. Use --interactive or --test_samples")

if __name__ == "__main__":
    main() 
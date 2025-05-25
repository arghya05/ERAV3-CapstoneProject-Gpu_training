#!/usr/bin/env python3
"""
Ultra-Fast Inference for Unsloth-trained LLaMA 3B Instruct
Optimized for speed and memory efficiency
"""

import torch
from unsloth import FastLanguageModel
import argparse

class UnslothTravelAssistantInference:
    def __init__(self, model_path, max_seq_length=2048):
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the Unsloth-trained model"""
        print(f"🚀 Loading Unsloth model from: {self.model_path}")
        
        # Load model with Unsloth for fast inference
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )
        
        # Enable fast inference mode
        FastLanguageModel.for_inference(self.model)
        
        print("✅ Model loaded successfully for fast inference!")
    
    def generate_response(self, query, max_new_tokens=512, temperature=0.7):
        """Generate response using Unsloth optimizations"""
        
        # Format using LLaMA 3 Instruct template
        messages = [
            {"role": "system", "content": "You are a helpful travel assistant specializing in India-centric travel planning. You provide detailed, culturally-aware advice for both domestic Indian travel and international destinations for Indian travelers."},
            {"role": "user", "content": query}
        ]
        
        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        # Generate with optimizations
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return response.strip()
    
    def interactive_mode(self):
        """Run interactive chat mode"""
        print("\n🌟 Welcome to your Ultra-Fast India Travel Assistant!")
        print("Powered by Unsloth-optimized LLaMA 3 Instruct")
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
                response = self.generate_response(query, max_new_tokens=300)
                print(response)
            except Exception as e:
                print(f"Error: {e}")
            
            print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Test Unsloth-trained LLaMA 3B Travel Assistant")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the Unsloth-trained model")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--test_samples", action="store_true", default=True,
                        help="Test with sample queries")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = UnslothTravelAssistantInference(args.model_path, args.max_seq_length)
    
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
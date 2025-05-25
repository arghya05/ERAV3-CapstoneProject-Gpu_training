#!/usr/bin/env python3
"""
Dataset validation script to check format and quality
"""

import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

def validate_dataset(file_path):
    """Validate dataset format and provide statistics"""
    
    print(f"\n=== Validating {file_path} ===")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    print(f"✅ Successfully loaded {len(data)} conversations")
    
    # Validate structure
    issues = []
    conversation_lengths = []
    user_message_lengths = []
    assistant_message_lengths = []
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            issues.append(f"Item {i}: Not a dictionary")
            continue
        
        if 'conversations' not in item:
            issues.append(f"Item {i}: Missing 'conversations' key")
            continue
        
        conversations = item['conversations']
        if not isinstance(conversations, list):
            issues.append(f"Item {i}: 'conversations' is not a list")
            continue
        
        conversation_lengths.append(len(conversations))
        
        for j, conv in enumerate(conversations):
            if not isinstance(conv, dict):
                issues.append(f"Item {i}, conversation {j}: Not a dictionary")
                continue
            
            if 'from' not in conv or 'value' not in conv:
                issues.append(f"Item {i}, conversation {j}: Missing 'from' or 'value' key")
                continue
            
            if conv['from'] not in ['user', 'assistant']:
                issues.append(f"Item {i}, conversation {j}: Invalid 'from' value: {conv['from']}")
                continue
            
            message_length = len(conv['value'])
            if conv['from'] == 'user':
                user_message_lengths.append(message_length)
            else:
                assistant_message_lengths.append(message_length)
    
    # Print validation results
    if issues:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"   - {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more issues")
        return False
    else:
        print("✅ Dataset format is valid!")
    
    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total conversations: {len(data)}")
    print(f"   Average conversation length: {sum(conversation_lengths) / len(conversation_lengths):.1f} messages")
    print(f"   Total user messages: {len(user_message_lengths)}")
    print(f"   Total assistant messages: {len(assistant_message_lengths)}")
    
    if user_message_lengths:
        print(f"   User message stats:")
        print(f"     - Average length: {sum(user_message_lengths) / len(user_message_lengths):.1f} chars")
        print(f"     - Min length: {min(user_message_lengths)} chars")
        print(f"     - Max length: {max(user_message_lengths)} chars")
    
    if assistant_message_lengths:
        print(f"   Assistant message stats:")
        print(f"     - Average length: {sum(assistant_message_lengths) / len(assistant_message_lengths):.1f} chars")
        print(f"     - Min length: {min(assistant_message_lengths)} chars")
        print(f"     - Max length: {max(assistant_message_lengths)} chars")
    
    return True

def create_dataset_visualization(train_path, val_path):
    """Create visualizations of the dataset"""
    
    def extract_stats(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversation_lengths = []
        user_lengths = []
        assistant_lengths = []
        
        for item in data:
            conversations = item['conversations']
            conversation_lengths.append(len(conversations))
            
            for conv in conversations:
                length = len(conv['value'])
                if conv['from'] == 'user':
                    user_lengths.append(length)
                else:
                    assistant_lengths.append(length)
        
        return conversation_lengths, user_lengths, assistant_lengths
    
    # Extract statistics
    train_conv_len, train_user_len, train_asst_len = extract_stats(train_path)
    val_conv_len, val_user_len, val_asst_len = extract_stats(val_path)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Dataset Analysis', fontsize=16)
    
    # Conversation length distribution
    axes[0, 0].hist([train_conv_len, val_conv_len], bins=20, alpha=0.7, 
                    label=['Train', 'Validation'], color=['blue', 'orange'])
    axes[0, 0].set_title('Conversation Length Distribution')
    axes[0, 0].set_xlabel('Number of Messages per Conversation')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # User message length distribution
    axes[0, 1].hist([train_user_len, val_user_len], bins=50, alpha=0.7,
                    label=['Train', 'Validation'], color=['blue', 'orange'])
    axes[0, 1].set_title('User Message Length Distribution')
    axes[0, 1].set_xlabel('Message Length (characters)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Assistant message length distribution
    axes[1, 0].hist([train_asst_len, val_asst_len], bins=50, alpha=0.7,
                    label=['Train', 'Validation'], color=['blue', 'orange'])
    axes[1, 0].set_title('Assistant Message Length Distribution')
    axes[1, 0].set_xlabel('Message Length (characters)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Dataset size comparison
    sizes = [len(train_conv_len), len(val_conv_len)]
    labels = ['Training', 'Validation']
    axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Dataset Split')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Dataset visualization saved as 'dataset_analysis.png'")

def main():
    """Main validation function"""
    
    print("🔍 Dataset Validation Tool")
    print("=" * 50)
    
    train_path = "./final/train_llama_format.json"
    val_path = "./final/validation_llama_format.json"
    
    # Validate both datasets
    train_valid = validate_dataset(train_path)
    val_valid = validate_dataset(val_path)
    
    if train_valid and val_valid:
        print("\n🎉 All datasets are valid!")
        
        # Create visualizations
        try:
            create_dataset_visualization(train_path, val_path)
        except Exception as e:
            print(f"⚠️  Could not create visualizations: {e}")
        
        print("\n✅ Your datasets are ready for training!")
        print("   Run: python train_llama3.py")
    else:
        print("\n❌ Please fix the dataset issues before training.")

if __name__ == "__main__":
    main() 
#!/bin/bash

echo "🚀 Llama 3 8B Travel Assistant - GPU Quick Start"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "train_llama3_gpu.py" ]; then
    echo "❌ Error: train_llama3_gpu.py not found!"
    echo "Make sure you're in the gpu_training directory"
    exit 1
fi

# Check if datasets exist
if [ ! -f "final/train_llama_format.json" ]; then
    echo "❌ Error: Training dataset not found!"
    echo "Make sure final/train_llama_format.json exists"
    exit 1
fi

if [ ! -f "final/validation_llama_format.json" ]; then
    echo "❌ Error: Validation dataset not found!"
    echo "Make sure final/validation_llama_format.json exists"
    exit 1
fi

echo "✅ All required files found"

# Check GPU availability
echo ""
echo "🔍 Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "🎯 Starting training..."
echo "This will take 3-20 minutes depending on your GPU"
echo ""

# Start training
python train_llama3_gpu.py

echo ""
echo "🎉 Training complete!"
echo "Your model is saved in the llama3_travel_assistant_* directory"
echo ""
echo "🧪 To test your model, run:"
echo "python test_model.py" 
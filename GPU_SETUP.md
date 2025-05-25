# 🚀 GPU Training Setup Guide

## 📁 What's in This Folder

This folder contains everything you need to train your Llama 3 8B Travel Assistant on GPU:

```
gpu_training/
├── train_llama3_gpu.py           # 👈 MAIN GPU TRAINING SCRIPT
├── test_model.py                 # Test your trained model
├── validate_dataset.py           # Validate your data
├── requirements.txt              # Python dependencies
├── .env                          # HuggingFace token (check this!)
├── README.md                     # Full documentation
├── setup.py                      # Package setup
├── LICENSE                       # MIT license
├── .gitignore                    # Git ignore rules
├── GPU_SETUP.md                  # This file
└── final/                        # Your datasets
    ├── train_llama_format.json   # 75,238 training samples
    └── validation_llama_format.json # 8,360 validation samples
```

## 🎯 Quick GPU Training (3 Steps)

### Step 1: Upload This Folder
Upload the entire `gpu_training/` folder to your GPU environment (Google Colab, RunPod, AWS, etc.)

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Start Training
```bash
python train_llama3_gpu.py
```

**That's it! Training will complete in 3-5 minutes on a good GPU.**

## 🖥️ Recommended GPU Environments

### Option 1: Google Colab Pro ($10/month)
- **GPU**: T4, V100, or A100
- **Setup**: Upload folder → Install deps → Run
- **Cost**: ~$0.50-1.00 per training run

### Option 2: RunPod ($0.20-0.80/hour)
- **GPU**: RTX 4090, A6000, A100
- **Setup**: Create pod → Upload → Run
- **Cost**: ~$0.20-2.00 per training run

### Option 3: AWS/Azure/GCP
- **GPU**: Various options
- **Setup**: Launch instance → Upload → Run
- **Cost**: Varies by provider

## ⚙️ GPU Training Settings

The GPU script (`train_llama3_gpu.py`) is optimized with:

- **Batch Size**: 4 (vs 1 on CPU)
- **Training Steps**: 500 (vs 5 on CPU)
- **Sequence Length**: 2048 (vs 512 on CPU)
- **FP16**: Enabled for speed
- **Gradient Accumulation**: 8 steps
- **Effective Batch Size**: 32 (4 × 8)

## 📊 Expected Results

### Training Progress:
```
=== Llama 3 8B Travel Assistant (GPU Version) ===
✅ GPU Available: NVIDIA A100-SXM4-40GB
GPU Memory: 40.0 GB
Loading datasets...
Training samples: 75238
Validation samples: 8360
Loading Llama 3 8B Instruct model for GPU...
trainable params: 41,943,040 || all params: 8,072,204,288 || trainable%: 0.5196
Starting GPU training...
Step 10/500: train_loss=1.234, eval_loss=1.456
Step 50/500: train_loss=0.987, eval_loss=1.123
...
Training Complete! Model saved to: ./llama3_travel_assistant_20240125_143022
```

### Training Time by GPU:
- **T4**: ~15-20 minutes
- **V100**: ~8-12 minutes  
- **A100**: ~3-5 minutes
- **RTX 4090**: ~5-8 minutes

## 🔧 Troubleshooting

### "CUDA out of memory"
Reduce batch size in `train_llama3_gpu.py`:
```python
per_device_train_batch_size=2,  # Reduce from 4 to 2
```

### "No GPU detected"
The script will warn you and fall back to CPU (very slow).

### "Access denied" for model
Check your `.env` file has a valid HuggingFace token.

### Training too slow
Make sure you're using a GPU instance, not CPU.

## ✅ Success Checklist

After training completes, you should have:

✅ **Model Directory**: `llama3_travel_assistant_YYYYMMDD_HHMMSS/`  
✅ **Low Loss**: Final training loss < 1.0  
✅ **Model Files**: `adapter_model.safetensors`, `adapter_config.json`  
✅ **Tokenizer**: `tokenizer.json`, `tokenizer_config.json`  

## 🧪 Test Your Model

```bash
python test_model.py
```

Example conversation:
```
User: I need help planning a trip to Tokyo
Assistant: I'd be happy to help you plan your trip to Tokyo! 
Let me gather some information to create the perfect itinerary for you.
What time of year are you planning to visit?
```

## 📦 Download Your Trained Model

After training, download the entire model directory:
- `llama3_travel_assistant_YYYYMMDD_HHMMSS/`

This contains your complete fine-tuned travel assistant!

## 🚀 Next Steps

1. **Test thoroughly** with various travel queries
2. **Deploy** to your application
3. **Share** with your team
4. **Iterate** - retrain with more data if needed

---

**Need help?** Check the main README.md for detailed troubleshooting! 
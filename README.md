# TinyLLM

A minimal transformer-based language model with only **626K parameters** that can hold coherent English conversations. Designed for CPU-only training and inference.

## Features

- **Ultra-compact**: 626,560 parameters (~2.4MB model file)
- **CPU-friendly**: No GPU required for training or inference
- **Character-level**: Simple tokenization with 59 characters
- **Fast inference**: Real-time response generation
- **Self-contained**: No external tokenizer dependencies

## Architecture

| Component | Value |
|-----------|-------|
| Parameters | 626,560 |
| Embedding Dim | 128 |
| Layers | 3 |
| Attention Heads | 4 |
| FFN Dim | 128 |
| Context Window | 128 chars |
| Vocab Size | 59 |

## Quick Start

### Requirements

```bash
pip install torch
```

### Run Chat Interface

```bash
python chat.py
```

### Train From Scratch

```bash
# 1. Generate dataset
python save_dataset.py

# 2. Train model (~20-30 min on CPU)
python train_final.py

# 3. Chat with your model
python chat.py
```

## Files

| File | Description |
|------|-------------|
| `model.py` | TinyLLM transformer architecture |
| `save_dataset.py` | Generates training dialogue dataset |
| `train_final.py` | Training script (8 epochs) |
| `chat.py` | Interactive chat interface |
| `tiny_llm.pt` | Pre-trained model weights |
| `dialogue.txt` | Training data (~1.2MB) |

## Example Conversation

```
You: Hello!
Bot: Hi there! How can I help you today?

You: What is AI?
Bot: AI stands for Artificial Intelligence. It refers to computer systems 
     designed to perform tasks that typically require human intelligence.

You: Tell me a joke
Bot: Why did the computer go to the doctor? Because it had a virus!
```

## Model Details

### Tokenization

Character-level tokenization mapping 59 unique characters:
- Lowercase letters (a-z)
- Uppercase letters (A-Z)  
- Digits (0-9)
- Punctuation and special tokens

### Training

- Optimizer: AdamW (lr=0.003, weight_decay=0.01)
- Epochs: 8
- Batch Size: 32
- Loss: Cross-entropy
- Final Loss: ~0.12

## Limitations

- Small context window (128 characters)
- Limited vocabulary and knowledge
- Best for short, simple conversations
- Memorization-based (small dataset)

## License

MIT License

## Acknowledgments

Built as an experiment in minimal viable language models. Demonstrates that coherent dialogue is achievable with sub-1M parameters through careful architecture design and training.

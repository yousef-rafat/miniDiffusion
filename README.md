# miniDiffusion

![SD3 Diagram](assets/display.png)

miniDiffusion is a reimplementation of the Stable Diffusion 3.5 model in pure PyTorch with minimal dependencies. It's designed for educational, experimenting, and hacking purposes.
It's made with the mindset of having the least amount of code necessary to recreate Stable Diffusion 3.5 from scratch, with only ~2800 spanning from VAE to DiT to the Train and Dataset scripts.

The main Stable Diffusion model code is located in dit.py, dit_components.py, and attention.py. The dit.py file is where the main model is located, dit_components.py is where the embedding, normalization, patch embedding, and help functions for the DiT code are located. The attention.py is where the Joint Attention implementation is located.
The noise.py is where the Euler Scheduler is located for solving the ODE of Rectified Flow. 

The t5_encoder.py and clip.py is where the text encoders are at, and their tokenizers are both located in tokenizer.py. The metrics.py is an implementation of the Fréchet inception distance (FID).

The common.py is a place for helper functions for training, the common_ds.py is an implementation of an iterable dataset that converts image data to trainable data for the DiT model.

> ⚠️ **Warning**:
> This repository still has experimental features and requires more testing.

## Components

### Core Image Generation Modules
- Implementations of VAE, CLIP, and T5 Text Encoders
- Implementation of Byte-Pair & Unigram tokenizers

### SD3 Components
- Multi-Modal Diffusion Transformer Model
- Flow-Matching Euler Scheduler
- Logit-Normal Sampling
- Joint Attention 

### Train and Inference Scripts For SD3

## Getting Started

Get the repo

```bash
git clone "https://github.com/yousef-rafat/miniDiffusion"
```

Install Dependencies
```bash
pip install -r requirements.txt
```

Install Checkpoints for Models
```bash
python3 encoders/get_checkpoints.py
```

# License

This project is under the MIT License and is made for educational and experimental purposes. 

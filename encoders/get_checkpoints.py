# Get Checkpoints to load for VAE, CLIP, and T5
# ////////////////////////// NOT MEANT TO BE RAN WHILE INFERENCE OR TRAINING ////////////////////////////////////////////////////////////////////////

from transformers import CLIPModel, T5EncoderModel, T5TokenizerFast
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def get_path(name: str):
    return os.path.join(os.getcwd(), "encoders", "hub", "checkpoints", name)

vae_repo = "stabilityai/sd-vae-ft-mse"
vae_cache = get_path("vae")

clip_repo = "openai/clip-vit-base-patch32"
clip_cache = get_path("clip")

t5_repo = "google/t5-v1_1-base"
t5_cache = get_path("t5")

vae_filename = "diffusion_pytorch_model.safetensors"
clip_filename = "pytorch_model.bin"

vae_weights_path = hf_hub_download(repo_id=vae_repo, filename = vae_filename, cache_dir = vae_cache)
print(f"VAE weights downloaded: {vae_weights_path}")

vae_state_dict = load_file(vae_weights_path)
torch.save(vae_state_dict, get_path("vae_checkpoint.pth"))

print("Converted and Saved VAE .safetensors to .pth")

# --- CLIP Part using Transformers ---

clip_model = CLIPModel.from_pretrained(clip_repo)
clip_model_state_dict = clip_model.state_dict()

torch.save(clip_model_state_dict,  get_path("clip_model.pth"))
print("Saved CLIP state_dict to .pth")

t5_encoder = T5EncoderModel.from_pretrained(t5_repo, cache_dir = t5_cache)
t5_encoder_state_dict = t5_encoder.state_dict()

torch.save(t5_encoder_state_dict, get_path("t5_encoder.pth"))
print("Saved T5 encoder state_dict to .pth")

tokenizer = T5TokenizerFast.from_pretrained("google/t5-v1_1-base")

save_dir = get_path("t5_tokenizer")
os.makedirs(save_dir, exist_ok=True)
tokenizer.save_pretrained(save_dir)

print(f"Saved tokenizer.json and related files to {save_dir}")
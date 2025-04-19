# Get Checkpoints to load for VAE and CLIP
# ////////////////////////// NOT MEANT TO BE RAN WHILE INFERENCE OR TRAINING ////////////////////////////////////////////////////////////////////////

from transformers import CLIPModel
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
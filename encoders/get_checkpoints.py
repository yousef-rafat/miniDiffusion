# Get Checkpoints to load for VAE, CLIP, T5, and SD3
# ////////////////////////// NOT MEANT TO BE RAN WHILE INFERENCE OR TRAINING ////////////////////////////////////////////////////////////////////////

from transformers import CLIPTextModelWithProjection, T5EncoderModel
from diffusers import StableDiffusion3Pipeline
from diffusers import SD3Transformer2DModel
import torch
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def get_path(name: str):
    return os.path.join(os.getcwd(), "encoders", "hub", "checkpoints", name)

vae_cache = get_path("vae")

clip_repo = "openai/clip-vit-large-patch14"
clip_cache = get_path("clip")

clip_2_repo = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
clip_2_cache = get_path("clip_2")

t5_repo = "google/t5-v1_1-xxl"
t5_cache = get_path("t5")

vae_filename = "diffusion_pytorch_model.safetensors"
clip_filename = "pytorch_model.bin"

diff_repo = "stabilityai/stable-diffusion-3.5-medium"
diff_filename = get_path("sd3")

if not os.path.exists(vae_cache):
    pipe = StableDiffusion3Pipeline.from_pretrained(diff_repo, token = "", torch_dtype = torch.bfloat16)
    vae_weights_path = pipe.vae.state_dict()

    torch.save(vae_weights_path, get_path("vae.pth"))

if not os.path.exists(diff_filename):

    model = SD3Transformer2DModel.from_pretrained(
        diff_repo,  subfolder="transformer",
        token = "",
        torch_dtype = torch.bfloat16
    )
    torch.save(model.to(torch.bfloat16).state_dict(), get_path("sd3_diff.pth"))

# --- CLIP Part using Transformers ---

if not os.path.exists(clip_cache):
    clip_model = CLIPTextModelWithProjection.from_pretrained(clip_repo, torch_dtype = torch.bfloat16)
    clip_model_state_dict = clip_model.state_dict()

    torch.save(clip_model_state_dict,  get_path("clip_model.pth"))
    print("Saved CLIP state_dict to .pth")

# must add token = "[TOKEN]" here
if not os.path.exists(clip_2_cache):
    clip_model = CLIPTextModelWithProjection.from_pretrained(clip_2_repo, cache_dir = clip_2_cache, torch_dtype = torch.bfloat16,
                                                         token = "")
    clip_model_state_dict = clip_model.state_dict()

    torch.save(clip_model_state_dict,  get_path("clip2.pth"))
    print(f"Saved CLIP state_dict to {clip_2_cache}")

if not os.path.exists(t5_cache):

    t5_encoder = T5EncoderModel.from_pretrained(t5_repo, cache_dir = t5_cache, torch_dtype = torch.bfloat16)
    t5_encoder_state_dict = t5_encoder.state_dict()

    torch.save(t5_encoder_state_dict, get_path("t5_encoder.pth"))
    print("Saved T5 encoder state_dict to .pth")
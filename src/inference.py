import os
import torch
from model.vae import VAE
import matplotlib.pyplot as plt
from model.dit import DiT, load_dit
from model.noise import NoiseScheduler
from model.dit_components import HandlePrompt
from model.clip import CLIP, load_clip, OpenCLIP
from model.t5_encoder import T5EncoderModel, load_t5

def denormalize(image):
    image = (image * 0.5 + 0.5).clamp(0, 1)

def inference(prompt: str, num_inference_steps: int = 50, device: str = "cpu"):
    """ Takes a Prompt and plots an image of what's generated """

    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    vae = VAE()

    model = DiT().eval()

    model_path = os.path.exists(os.getcwd(), "model", "checkpoint.pth")

    if model_path:
        model = model.load_state_dict(model_path)
    else:
        model = load_dit(model)

    clip = CLIP()

    clip_2 = OpenCLIP()
    t5_encoder = T5EncoderModel()
    prompt_handler = HandlePrompt()

    noiser = NoiseScheduler(num_inference_timesteps = num_inference_steps, inference = True)

    timesteps = noiser.timesteps

    clip = load_clip(clip = clip, clip_2 = clip_2, device = device)
    t5_encoder = load_t5(model = t5_encoder, device = device)

    checkpoint_path = os.path.join(os.getcwd(), "model", "checkpoint")
    model.load_state_dict(checkpoint_path)

    latent = torch.randn(1, 16, 48, 48, device = model.device)  # initial latent space

    embeddings, pooled_projections = prompt_handler(prompt, clip = clip, clip_2 = clip_2, t5_encoder = t5_encoder)
    timesteps = noiser.timesteps

    # inference loop
    with torch.no_grad():
        for t in timesteps:

            timestep = t.expand(latent.size(0)) # scale to batch size

            outputs = model(latent, encoder_hidden_states = embeddings, pooled_projections = pooled_projections, timestep = timestep)

            latent = noiser.reverse_flow(current_sample = latent, model_output = outputs, timestep = timestep, stochasticity = False)

    
    latent = (latent / vae.scaling_factor) + vae.shift_factor
    generated_image = vae.decode(latent)
    generated_image = denormalize(generated_image)

    plt.figure()
    plt.imshow(generated_image.permute(1, 2, 0).detach().to(torch.float32).numpy()) 
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    inference("A pair of black shoes", num_inference_steps = 50)
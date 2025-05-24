import os
import torch
from model.dit import DiT
from model.vae import VAE
import matplotlib.pyplot as plt
from model.noise import NoiseScheduler
from model.dit_components import HandlePrompt
from model.clip import CLIP, load_clip, OpenCLIP
from model.t5_encoder import T5EncoderModel, load_t5

def inference(prompt: str, num_inference_steps: int = 50, device: str = "cpu"):
    """ Takes a Prompt and plots an image of what's generated """

    model = DiT().to(device).eval()

    vae = VAE()
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

    latent = torch.randn(1, 4, 32, 32, device = model.device)  # initial latent space

    embeddings, pooled_projections = prompt_handler(prompt, clip = clip, clip_2 = clip_2, t5_encoder = t5_encoder)

    # inference loop
    with torch.no_grad():
        for t in timesteps:

            timestep = t.expand(latent.size(0)) # scale to batch size

            outputs = model(latent, encoder_hidden_states = embeddings, pooled_projections = pooled_projections, timestep = timestep)

            latent = noiser.reverse_flow(current_sample = latent, model_output = outputs, timestep = timestep)

    generated_image = vae.decode(latent)

    plt.figure()
    plt.imshow(generated_image.permute(1, 2, 0).detach().to(torch.float32).numpy()) 
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    inference("A pair of black shoes", num_inference_steps = 50)
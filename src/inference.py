import os
import torch
from model.dit import DiT
from model.vae import VAE
import matplotlib.pyplot as plt
from model.noise import NoiseScheduler
from model.clip import CLIP, load_clip

def inference(prompt: str, num_inference_steps: int = 50, device: str = "cpu"):
    """ Takes a Prompt and plots an image of what's generated """

    model = DiT(depth = 10).to(device).eval()
    vae = VAE()
    clip = CLIP()
    noiser = NoiseScheduler(beta = 0.9, timesteps = 10)

    sigmas = noiser.sigma_scheduler(num_inference_steps)

    clip = load_clip(clip)
    input_ids, attention_mask = clip.encode_text(prompt)

    checkpoint_path = os.path.join(os.getcwd(), "model", "checkpoint")
    model.load_state_dict(checkpoint_path)

    latent = torch.randn(1, 4, 32, 32, device = model.device)  # initial latent space

    # inference loop
    with torch.no_grad():
        for i in range(len(sigmas) - 1):

            sigma = sigmas[i]
            next_sigma = sigmas[i + 1]

            t = torch.tensor([sigma], device = model.device)

            outputs = model(latent, input_ids = input_ids, attention_mask = attention_mask, t = t)
            # solver_fn could be rk4 or euler
            velocity = noiser.rk4_solver(outputs, t)

            x_pred = latent - sigma * velocity

            noise = torch.randn_like(latent)

            x = (1 - next_sigma) * x_pred + next_sigma * noise

    generated_image = vae.decode(x)

    plt.figure()
    plt.imshow(generated_image.permute(1, 2, 0).detach().to(torch.float32).numpy()) 
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    inference("A pair of black shoes", num_inference_steps = 50)
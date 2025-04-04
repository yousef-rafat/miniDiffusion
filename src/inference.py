import os
import torch
from model.dit import DiT
from model.vae import VAE
import matplotlib.pyplot as plt
from model.noise import NoiseScheduler
from model.dit_components import HandlePrompt

def inference(prompt: str):
    """ Takes a Prompt and plots an image of what's generated """

    model = DiT(depth = 10)
    vae = VAE()
    noiser = NoiseScheduler(beta = 0.9, timesteps = 10)

    handle_prompt = HandlePrompt()
    input_ids, attention_mask = handle_prompt(prompt)

    checkpoint_path = os.path.join(os.getcwd(), "model", "checkpoint")
    model.load_state_dict(checkpoint_path)

    # solver_fn could be rk4 or euler
    generated_latent = model.generate(noiser.rk4_solver, input_ids = input_ids, attention_mask = attention_mask)
    generated_image = vae.decode(generated_latent)

    plt.figure()
    plt.imshow(generated_image.permute(1, 2, 0).detach().to(torch.float32).numpy()) 
    plt.show()

if __name__ == "__main__":
    prompt = "Black Shoes"
    inference(prompt)
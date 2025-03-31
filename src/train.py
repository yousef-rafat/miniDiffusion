import os
import torch
import argparse
from model.dit import DiT
from model.vae import VAE
from model.clip import CLIP
from typing import Optional
from model.metrics import FID
from model.noise import NoiseScheduler
from common_ds import get_dataset, get_fashion_mnist_dataset, check_dataset_dir
from common import loss_fn, interpolate_samples, get_ground_truth_velocity, compute_clip_loss

def train(device: Optional[str], train_dataset: Optional[str], eval_dataset: Optional[str], epochs: int = 10, lr: float = 0.003, batch_size: int = 64, 
          euler: bool = False, log: bool = False):
    
    model = DiT(embedding_size = 512, heads = 8, depth = 6).to(device)

    # turn fused only on when gpu is available
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr,
                                  fused = True if torch.cuda.is_available() else False)

    noiser = NoiseScheduler(beta = 0.9, timesteps = 10)
    metric = FID()
    vae = VAE()
    clip = CLIP()

    save_path = os.path.join(os.getcwd(), "model")

    if not device: device = next(model.parameters()).device

    if not train_dataset or not eval_dataset: 
        print("Defaulting to Fashion MNIST")
        train_dataset, eval_dataset = get_fashion_mnist_dataset(batch_size = batch_size, device = device)

    elif train_dataset:
        check_dataset_dir("train_ds")
        train_dataset = get_dataset(train_dataset, batch_size = batch_size, device = device)

    elif eval_dataset:
        check_dataset_dir("eval_ds")
        eval_dataset = get_dataset(eval_dataset, batch_size = batch_size, device = device)

    alpha_bar = noiser.get_alpha

    def fit(image, noise, input_ids, attention_mask):

        # remove extra dimension
        noise = noise.squeeze(1)
        input_ids = input_ids.long().squeeze(2)

        # random times for training
        # broadcast height, width, and channels
        t = torch.rand(image.size(0), 1, 1, 1).to(device)

        # scale to timesteps and turn to long
        t_index = (t * (noiser.timesteps - 1)).long()
        alpha_t = alpha_bar[t_index]

        intr_sampls = interpolate_samples(image, noise, t)
        directions = get_ground_truth_velocity(image, noise, alpha_t)

        drift = model(latent = intr_sampls, t = t, input_ids = input_ids, attention_mask = attention_mask)
        loss = loss_fn(drift, directions, alpha_t = alpha_t)

        if euler: denoised_latent = noiser.euler_solver(model, noise)
        else: denoised_latent = noiser.rk4_solver(model, noise)

        generated_image = vae.decode(denoised_latent)

        clip_loss = compute_clip_loss(clip, generated_image, input_ids, attention_mask, size = generated_image.size(-1))
        loss += clip_loss

        return loss, generated_image

    train_losses, eval_losses = [], []
    for epoch in range(epochs):

        total_loss = 0
        model.train()

        print("Starting to train...")
        for i, (image, noise, input_ids, attention_mask) in enumerate(train_dataset):
            
            optimizer.zero_grad()

            loss, generated_image = fit(image = image, noise = noise, input_ids = input_ids, attention_mask = attention_mask)

            fid_score = metric(generated_image, image)

            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)

            if log: train_losses.append(avg_loss)

            loss.backward()
            optimizer.step()

            description = f'Train Epoch: {epoch}, Loss: {avg_loss}, FID: {fid_score}'
            print(description)

        if not eval_dataset: continue

        model.eval()
        eval_loss = 0
        for i, (image, noise, input_ids, attention_mask) in enumerate(eval_dataset):
            loss, generated_image = fit(image = image, noise = noise, input_ids = input_ids, attention_mask = attention_mask)
            fid_score = metric(generated_image, image)

            eval_loss += loss.item()
            avg_loss = eval_loss / (i + 1)

            if log: eval_losses.append(avg_loss)

        description = f'Eval Epoch: {epoch}, Loss: {avg_loss}, FID: {fid_score}'
        print(description)

    # save model and log the output
    torch.save(model.state_dict(), save_path)

    if log:
        import csv
        with open("log.csv", "w", newline = "") as f:
            writer = csv.writer(f)
            writer.writerow(["Train", "Eval"])
            writer.writerows(zip(train_losses, eval_losses))


def get_args():
    parser = argparse.ArgumentParser(description = "Train a model with specified parameters.")

    parser.add_argument("--device", type = str, choices=["cpu", "cuda"], default = "cpu", help = "Device to use (cpu or cuda)")
    parser.add_argument("--train_dataset", type = str, required = False, help = "Path to the training dataset")
    parser.add_argument("--eval_dataset", type = str, required = False, help = "Path to the evaluation dataset")
    parser.add_argument("--epochs", type = int, default = 5, help = "Number of training epochs")
    parser.add_argument("--lr", type = float, default = 0.003, help = "Learning rate")
    parser.add_argument("--batch_size", type = int, default = 2, help = "Batch size for training")
    parser.add_argument("--euler", action = "store_true", help=  "Enable Euler mode")
    parser.add_argument("--log", action = "store_true", help = "Enable logging")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    train(
        device = args.device,
        train_dataset = args.train_dataset,
        eval_dataset = args.eval_dataset,
        epochs = args.epochs,
        lr = args.lr,
        batch_size = args.batch_size,
        euler = args.euler,
        log = args.log
    )
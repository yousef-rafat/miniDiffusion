import os
import torch
import argparse
from model.dit import DiT
from typing import Optional
from model.metrics import FID
from model.vae import VAE, load_vae
from model.clip import CLIP, load_clip
from model.noise import NoiseScheduler
from torch.optim.lr_scheduler import LambdaLR
from common_ds import get_dataset, get_fashion_mnist_dataset, check_dataset_dir
from common import loss_fn, interpolate_samples, get_ground_truth_velocity, compute_clip_loss

def train(device: Optional[str], train_dataset: Optional[str], eval_dataset: Optional[str] = None, epochs: int = 3, lr: float = 0.0001, batch_size: int = 64, 
          euler: bool = False, log: bool = True):

    model = DiT(embedding_size = 512, heads = 8, depth = 10).to(device = device)

    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(2025)

    # turn fused only on when gpu is available
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr,
                                  fused = True if torch.cuda.is_available() else False)
    
    # go from 0 to inital learning (lr) in warmup_steps and stay there
    warmup_steps = 100
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)

    noiser = NoiseScheduler(beta = 0.9, timesteps = 10)
    metric = FID()
    vae = VAE()
    clip = CLIP()

    vae = load_vae(model = vae, device = device)
    clip = load_clip(model = clip, device = device)

    save_path = os.path.join(os.getcwd(), "model", "checkpoint.pth")
    if os.path.exists(save_path): model.load_state_dict(torch.load(save_path))

    if not device: device = next(model.parameters()).device

    if not train_dataset or not eval_dataset: 
        print("Defaulting to Fashion MNIST")
        train_dataset, eval_dataset = get_fashion_mnist_dataset(batch_size = batch_size, device = device)

    elif train_dataset:
        check_dataset_dir("train_ds")
        train_dataset = get_dataset(train_dataset, batch_size = batch_size, device = device)

    elif eval_dataset is not None:
        check_dataset_dir("eval_ds")
        eval_dataset = get_dataset(eval_dataset, batch_size = batch_size, device = device)

    alpha_bar = noiser.get_alpha

    def fit(image, noise, input_ids, attention_mask):

        # remove extra dimension
        noise = noise.squeeze(1)
        input_ids = input_ids.squeeze(2)

        # random times for training
        # broadcast height, width, and channels
        t = torch.rand(image.size(0), 1, 1, 1).to(device)

        # scale to timesteps and turn to long
        t_index = (t * (noiser.timesteps - 1)).long()
        alpha_t = alpha_bar[t_index]

        intr_sampls = interpolate_samples(source = image, target = noise, t = t)
        directions = get_ground_truth_velocity(image, noise, alpha_t)

        drift = model(latent = intr_sampls, t = t, input_ids = input_ids, attention_mask = attention_mask)
        loss = loss_fn(drift, directions, alpha_t = alpha_t)
        
        if euler: denoised_latent = noiser.euler_solver(model, noise.detach())
        else: denoised_latent = noiser.rk4_solver(model, noise.detach())

        generated_image = vae.decode(denoised_latent.detach())

        clip_loss = compute_clip_loss(clip, generated_image, input_ids, attention_mask, size = generated_image.size(-1))
        loss += clip_loss

        return loss, generated_image
    
    def stack_tensors(batch):

        # stack all the tensors to get elements of tuples
        latent = torch.stack([sample[0] for sample in batch]).squeeze(1)
        noise = torch.stack([sample[1] for sample in batch]).squeeze(1) 
        image = torch.stack([sample[2] for sample in batch]).squeeze(1)
        input_ids = torch.stack([sample[3] for sample in batch]).squeeze(1)
        attention_mask = torch.stack([sample[4] for sample in batch]).squeeze(1)

        return latent, noise, image, input_ids, attention_mask

    print("Starting to train...")
    train_losses, eval_losses = [], []
    for epoch in range(epochs):

        total_loss = 0
        model.train()

        # (latent, noise, image, input_ids, attention_mask)
        for i, batch in enumerate(train_dataset):
            
            latent, noise, image, input_ids, attention_mask = stack_tensors(batch)

            loss, generated_image = fit(image = latent, noise = noise, input_ids = input_ids, attention_mask = attention_mask)

            fid_score = metric(generated_image, image)

            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)

            if log: train_losses.append(avg_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # reset the KV cache to not get multiple (unintented) backward passes
            for layer in model.model:
                if hasattr(layer, "self_attn"):
                    layer.self_attn.reset_cache()

        description = f'Train Epoch: {epoch + 1}, Loss: {avg_loss:.2f}, FID: {fid_score:.2f}'
        print(description)

        if eval_dataset is None: continue

        model.eval()
        eval_loss = 0
        for i, batch in enumerate(eval_dataset):

            latent, noise, image, input_ids, attention_mask = stack_tensors(batch)

            with torch.no_grad():
                loss, generated_image = fit(image = latent, noise = noise, input_ids = input_ids, attention_mask = attention_mask)

            fid_score = metric(generated_image, image)

            eval_loss += loss.item()
            avg_loss = eval_loss / (i + 1)

            if log: eval_losses.append(avg_loss)

        description = f'Eval Epoch: {epoch + 1}, Loss: {avg_loss:.2f}, FID: {fid_score:.2f}'
        print(description)

    # save model and log the output
    torch.save(model.state_dict(), save_path)

    if log:
        import csv
        log_dir = os.path.join("model", "log.csv")
        with open(log_dir, "w", newline = "") as f:
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
    parser.add_argument("--batch_size", type = int, default = 4, help = "Batch size for training")
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
    )
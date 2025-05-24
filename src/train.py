import os
import torch
import argparse
from model.dit import DiT
from common import loss_fn
from typing import Optional
from model.vae import VAE, load_vae
from model.dit_components import HandlePrompt
from torch.optim.lr_scheduler import LambdaLR
from model.clip import CLIP, load_clip, OpenCLIP
from model.t5_encoder import T5EncoderModel, load_t5
from common_ds import get_dataset, get_fashion_mnist_dataset, check_dataset_dir

def train(device: Optional[str], train_dataset: Optional[str], epochs: int = 3, lr: float = 0.0001, batch_size: int = 8, 
          log: bool = True):
    
    """ In Training, we will only care about decreasing the loss to save computation """

    model = DiT().to(device = device)

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

    vae = VAE()
    clip = CLIP()

    clip_2 = OpenCLIP()
    t5_encoder = T5EncoderModel()
    prompt_handler = HandlePrompt()

    vae = load_vae(model = vae, device = device)
    clip = load_clip(model = clip, model_2 = clip_2, device = device)
    t5_encoder = load_t5(model = t5_encoder, device = device)

    save_path = os.path.join(os.getcwd(), "model", "checkpoint.pth")
    if os.path.exists(save_path): model.load_state_dict(torch.load(save_path))

    if not device: device = next(model.parameters()).device

    if not train_dataset: 
        print("Defaulting to Fashion MNIST")
        train_dataset, _ = get_fashion_mnist_dataset(batch_size = batch_size, device = device)

    elif train_dataset:
        check_dataset_dir("train_ds")
        train_dataset = get_dataset(train_dataset, batch_size = batch_size, device = device)
    
    def stack_tensors(batch):

        # stack all the tensors to get elements of tuples
        latent = torch.stack([sample[0] for sample in batch]).squeeze(1)
        noised_image = torch.stack([sample[1] for sample in batch]).squeeze(1) 
        added_noise = torch.stack([sample[2] for sample in batch]).squeeze(1)
        timesteps = torch.stack([sample[3] for sample in batch]).squeeze(1)
        timesteps = torch.stack([sample[4] for sample in batch]).squeeze(1)

        return (
            latent, 
            noised_image, 
            added_noise, 
            timesteps, 
            label
        )

    print("Starting to train...")

    model.train()
    train_losses, eval_losses = [], []
    for epoch in range(epochs):

        total_loss = 0
        # (latent, noise, image, input_ids, attention_mask)
        for i, batch in enumerate(train_dataset):
            
            latent, noised_image, added_noise, timesteps, label = stack_tensors(batch)

            embeddings, pooled_embeddings = prompt_handler(label, clip = clip, clip_2 = clip_2, t5_encoder = t5_encoder)

            target = added_noise - noised_image

            drift = model(latent = latent,
                          timestep = timesteps,
                          encoder_hidden_states = embeddings,
                          pooled_projections = pooled_embeddings)
        
            loss = loss_fn(drift, target)

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

        print(f'Train Epoch: {epoch + 1}, Loss: {avg_loss:.2f}')

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
    )
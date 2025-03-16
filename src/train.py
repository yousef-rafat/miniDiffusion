import os
import torch
from model.vae import VAE
from typing import Optional
from model.metrics import FID
from model.noise import NoiseScheduler
from torch.utils.data import Dataset # just for the type
from common_ds import get_dataset, get_fashion_mnist_dataset
from common import loss_fn, interpolate_samples, get_ground_truth_velocity, compute_clip_loss

def train(model, device: Optional[str], train_dataset: Optional[Dataset], epochs = 10, lr = 0.003, batch_size = 64, euler: bool = False,
          fashion: bool = True):

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, fused = True)

    noiser = NoiseScheduler(beta = 0.9)
    metric = FID()
    vae = VAE()

    if not device: device = next(model.parameters()).device

    if not train_dataset: 
        if fashion: get_fashion_mnist_dataset()
        else:
            train_path = os.path.join(os.getcwd(), "train_ds")
            assert os.path.exists(train_path), "train_ds should exist in working directory miniDiffusion or should be passed as a parameter"
            train_dataset = get_dataset(train_path, batch_size = batch_size)

    model.train()
    alpha_t = noiser.get_alpha

    for epoch in range(epochs):
        total_loss = 0
        for i, (image, noise, input_ids, attention_mask) in enumerate(train_dataset):
            
            optimizer.zero_grad()

            # random times for training
            t = torch.rand(image.size(0), 1).to(device)

            intr_sampls = interpolate_samples(image, noise, t)
            directions = get_ground_truth_velocity(image, noise, t, alpha_t)

            drift = model(intr_sampls, t)
            loss = loss_fn(drift, directions, alpha_t = alpha_t)

            if euler: denoised_latent = noiser.euler_solver(model, noise)
            else: denoised_latent = noiser.rk4_solver(model, noise)

            generated_image = vae.decode(denoised_latent)

            fid_score = metric(generated_image, image)
            clip_loss = compute_clip_loss(generated_image, input_ids, attention_mask)

            loss += clip_loss

            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)

            loss.backward()
            optimizer.step()

        description = f'Epoch: {epoch}, Loss: {avg_loss}, FID: {fid_score}'
        print(description)
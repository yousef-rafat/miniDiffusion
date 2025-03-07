import os
import torch
import torch.nn as nn
import torchvision.io as io
from torch.utils.data import Dataset
from model.noise import NoiseScheduler 

class OptimizeImage(nn.Module):
    " optimizes an image and makes it trainable "
    def __init__(self, image: torch.Tensor):
        super(OptimizeImage, self).__init__()
        self.opt_image = nn.Parameter(image.clone().detach())

    def forward(self):
        return self.opt_image
    
class ImageDataset(Dataset):
    def __init__(self, dataset_dir: str, transforms = None):
        super(ImageDataset, self).__init__()
        
        self.images = [os.path.join(dataset_dir, img_dir) for img_dir in os.listdir(dataset_dir)]
        self.transforms = transforms
        self.scheduler = NoiseScheduler()
        self.make_trainable = OptimizeImage()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # in the forward pass of the model
        # we compare both the distributions of noise and image

        image_dir = self.images[idx]
        image = io.read_image(image_dir)

        if self.transforms:
            image = self.transforms(image)

        # normalize
        image = image.float() / 255.0

        noise = self.scheduler.add_noise(image)

        noise = self.make_trainable(noise)

        return image, noise
    

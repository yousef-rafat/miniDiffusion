import os
import torch
import torch.nn as nn
import torchvision.io as io
from model.noise import NoiseScheduler 
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset, DataLoader

effects = v2.Compose([
    v2.RandomRotation(degrees=(0, 60)),
    v2.RandomHorizontalFlip(),
    v2.RandomPerspective(distortion_scale = 0.4, p = 0.6)
])

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
        # we compare both the distributions of noise and image (returned values)

        image_dir = self.images[idx]
        image = io.read_image(image_dir)

        if self.transforms:
            image = self.transforms(image)

        # normalize
        image = image.float() / 255.0

        noise = self.scheduler.add_noise(image)

        noise = self.make_trainable(noise)

        return image, noise
    
def get_dataset(path: str, batch_size: int, shuffle: bool = True) -> DataLoader:
    " Get dataset ready "
    dataset = ImageDataset(path, transforms = effects)
    dataloader = DataLoader(dataset, shuffle = shuffle, batch_size = batch_size)

    return dataloader

def get_fashion_mnist_dataset(batch_size: int = 32, shuffle: bool = True) -> DataLoader:

    train_dataset = torchvision.datasets.FashionMNIST(root = "./data", train = True, download = True, transform = effects)
    train_dataset = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size)

    return train_dataset
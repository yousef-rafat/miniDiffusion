import os
import torch
import shutil
import torchvision
import torch.nn as nn
from model.vae import VAE
import torchvision.io as io
import matplotlib.pyplot as plt
from model.noise import NoiseScheduler 
import torchvision.transforms.v2 as v2
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from model.dit_components import HandlePrompt
from torch.utils.data import Dataset, DataLoader

effects = v2.Compose([
    v2.RandomRotation(degrees=(0, 60)),
    v2.RandomHorizontalFlip(),
    v2.RandomPerspective(distortion_scale = 0.4, p = 0.6),
    ToTensor()
])

class OptimizeImage(nn.Module):
    " optimizes an image and makes it trainable "
    def __init__(self):
        super(OptimizeImage, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.tensor:
        self.opt_image = nn.Parameter(image.clone().detach())
        return self.opt_image

class NamedImageFolder(ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        # convert index to class name
        class_name = self.classes[label]
        return image, class_name
    
class ImageDataset(Dataset):
    def __init__(self, image_dataset: NamedImageFolder, transforms = None):
        super(ImageDataset, self).__init__()
        
        self.images = [image for image, _ in image_dataset.samples]
        self.labels = [image_dataset.classes[label] for _, label in image_dataset.samples]

        self.transforms = transforms
        self.scheduler = NoiseScheduler(beta = 0.9, timesteps = 10)
        self.make_trainable = OptimizeImage()
        self.prompt_handle = HandlePrompt()
        self.vae = VAE()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # in the forward pass of the model
        # we compare both the distributions of noise and image (returned values)
        # along with using input_ids and attention_mask for clip loss

        image_dir = self.images[idx]
        image = io.read_image(image_dir)

        label = self.labels[idx]
        input_ids, attention_mask = self.prompt_handle(label)

        if self.transforms:
            image = self.transforms(image)

        # normalize
        image = image.float() / 255.0
        image = image.expand(3, -1, -1) # turn to RGB

        _, _, latent = self.vae.encode(image) # turn to latent space

        noise, _ = self.scheduler.add_noise(latent.unsqueeze(0))

        noise = self.make_trainable(noise)

        noise = noise.detach()

        return latent.detach(), noise, input_ids.detach(), attention_mask.detach()

def check_dataset_dir(type: str):
    # check if the dir of dataset is valid
    path = os.path.join(os.getcwd(), type)
    assert os.path.exists(path), "{type} should exist in working directory miniDiffusion or should be passed as a parameter"
    
    return True

def get_dataloader(dataset, batch_size: int = 32, shuffle: bool = True, device: str = "cpu") -> DataLoader:
    """ Returns an optimized DataLoader based on the computing device. """

    # num of workers == num of cpu cores    
    num_workers = max(1, os.cpu_count() // 2) if device == "cuda" else max(1, os.cpu_count() // 4)
    
    # enable pin memory for gpu
    pin_memory = device == "cuda"
    
    prefetch_factor = 2 if num_workers > 0 else None
    persistent_workers = num_workers > 0
    
    return DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        pin_memory = pin_memory,
        persistent_workers = persistent_workers,
        prefetch_factor = prefetch_factor
    )

def get_dataset(path: str, batch_size: int, shuffle: bool = True, device: str = "cpu") -> DataLoader:
    " Get dataset ready "

    dataset = NamedImageFolder(path, transform = effects)
    dataset = ImageDataset(dataset)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)#get_dataloader(dataset, shuffle = shuffle, batch_size = batch_size, device = device)

    return dataloader

def get_fashion_mnist_dataset(batch_size: int = 32, shuffle: bool = True, device: str = "cpu") -> DataLoader:
    """
    Get Fashion MNIST dataset ready for train.py script
    """
    #  base directory
    base_dir = os.path.join(os.getcwd(), "data", "fashion_mnist")
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):

        train_dataset = torchvision.datasets.FashionMNIST(root="./data", train = True, download = True)
        test_dataset = torchvision.datasets.FashionMNIST(root="./data", train = False, download = True)

        class_names = train_dataset.classes 

        for class_name in class_names:
            os.makedirs(os.path.join(train_dir, class_name), exist_ok = True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok = True)

        def save_images(dataset, split):
            # save images into folder
            root_dir = train_dir if split == "train" else test_dir

            for idx in range(len(dataset)):
                img, label = dataset[idx]
                class_name = class_names[label]
                img_path = os.path.join(root_dir, class_name, f"{idx}.png")
                img.save(img_path)

        # Save train and test sets
        save_images(train_dataset, "train")
        save_images(test_dataset, "test")

        shutil.rmtree(os.path.join(os.getcwd(), "data", "FashionMNIST"))

    train_dataset = get_dataset(train_dir, batch_size = batch_size, shuffle = shuffle, device = device)
    test_dataset = get_dataset(test_dir, batch_size = batch_size, shuffle = shuffle, device = device)

    return train_dataset, test_dataset

def test_fashion():
    train_dataset, _ = get_fashion_mnist_dataset()

    for (image, _, label, _) in train_dataset:
        image = image[0]
        label = label[0]

        plt.figure()
        plt.imshow(image.permute(1, 2, 0).detach().numpy()) 
        plt.title(label)
        plt.show()
        break
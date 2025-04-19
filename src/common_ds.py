import os
import torch
import shutil
import torchvision
from PIL import Image
import torch.nn as nn
from model.vae import VAE
import matplotlib.pyplot as plt
from model.noise import NoiseScheduler 
import torchvision.transforms.v2 as v2
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from model.dit_components import HandlePrompt
from torch.utils.data import IterableDataset, DataLoader 

effects = v2.Compose([
    v2.RandomRotation(degrees=(0, 60)),
    v2.RandomHorizontalFlip(),
    v2.RandomPerspective(distortion_scale = 0.4, p = 0.6),
    v2.Resize(size = (256, 256)), # for vae
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
    
class ImageDataset(IterableDataset):
    def __init__(self, image_dataset: NamedImageFolder, transforms = None, batch: int = 4, max_samples = 12):
        super(ImageDataset, self).__init__()
        
        self.images = image_dataset.samples # list (img_path, label_index)
        self.labels = image_dataset.classes

        self.transforms = transforms
        self.scheduler = NoiseScheduler(beta = 0.9, timesteps = 10)
        self.make_trainable = OptimizeImage()
        self.prompt_handle = HandlePrompt()
        self.vae = VAE()

        self.batch_size = batch
        self.max_samples = max_samples

    def __iter__(self):
        # in the forward pass of the model
        # we compare both the distributions of noise and image (returned values)
        # along with using input_ids and attention_mask for clip loss

        batch = []
        sample_count = 0
        # lazy loading for efficient memory usuage
        for image_path, label_idx in self.images:

            try:

                if sample_count >= self.max_samples:
                    break

                image = Image.open(image_path)

                label = self.labels[label_idx]
                input_ids, attention_mask = self.prompt_handle(label)

                if self.transforms:
                    image = self.transforms(image)

                image = image.expand(3, -1, -1) # turn to RGB

                _, _, latent = self.vae.encode(image) # turn to latent space

                noise, _ = self.scheduler.add_noise(latent.unsqueeze(0))

                noise = self.make_trainable(noise)

                # do manual batching
                batch.append((latent,
                             noise,
                             image,
                             input_ids.detach(),
                             attention_mask.detach()))
                
                sample_count += 1
                
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

            except Exception as e:
                print("(-) Error: ", e)

def check_dataset_dir(type: str):
    # check if the dir of dataset is valid
    path = os.path.join(os.getcwd(), type)
    assert os.path.exists(path), "{type} should exist in working directory miniDiffusion or should be passed as a parameter"
    
    return True

def get_dataloader(dataset, device: str = "cpu") -> DataLoader:
    """ Returns an optimized DataLoader based on the computing device. """
    
    # enable pin memory for gpu
    pin_memory = device == "cuda"

    return DataLoader(
        dataset,
        batch_size = 1, # batch size one for IterableDataset
        num_workers = 0,
        pin_memory = pin_memory,
        prefetch_factor = None,
        persistent_workers = False,
    )

def get_dataset(path: str, batch_size: int, device: str = "cpu") -> DataLoader:
    " Get dataset ready "

    dataset = NamedImageFolder(path)
    dataset = ImageDataset(dataset, batch = batch_size, transforms = effects)
    dataloader = get_dataloader(dataset, device = device)

    return dataloader

def get_fashion_mnist_dataset(batch_size: int = 2, device: str = "cpu") -> DataLoader:
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

    train_dataset = get_dataset(train_dir, batch_size = batch_size, device = device)
    test_dataset = get_dataset(test_dir, batch_size = batch_size, device = device)

    return train_dataset, test_dataset

def test_fashion():
    train_dataset, _ = get_fashion_mnist_dataset()

    for batch in train_dataset: # (_, _, image, label, _)
        image = batch[0][2]
        label = batch[0][3]

        plt.figure()
        plt.imshow(image.squeeze(0).permute(1, 2, 0).detach().to(torch.float32).numpy()) 
        plt.title(label.squeeze(0))
        plt.show()
        break
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

def patchify(x: Tensor, size: int, stride = None):
    " Turn a latent into patches "
    # usually, the smaller the patch size the better quality of the generated image
    # if you have enough gpu power, try training with patch size of 2
    batch, channels, height, width = x.shape

    assert height % 4 == 0 and width % 4 == 0, "Image must be divisible into 16 patches"

    if stride is None: stride = size  # no overlap

    patches = x.unfold(2, size, stride).unfold(3, size, stride)

    # (batch, num_patches, channels, patch_size, patch_size)
    patches = patches.contiguous().view(batch, -1, channels, size, size)

    return patches

def depatchify(x: Tensor) -> torch.Tensor:
    " turn patches into an image "

    # check if image is 4 by 4 patches
    batch, patches, channels, size, _ = x.shape

    grid_size = math.ceil(math.sqrt(patches))
    total_patches = grid_size * grid_size
    img_size = grid_size * size

    # pad the tensor with zeros if there's not enough patches
    if patches < total_patches:

        pad_count = total_patches - patches
        pad_tensor = torch.zeros(batch, pad_count, channels, size, size, device = x.device, dtype = x.dtype)
        x = torch.cat([x, pad_tensor], dim = 1)

    # turning patches into images 
    x = x.view(batch, grid_size, grid_size, channels, size, size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(batch, channels, img_size, img_size) 

    return x
        

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len = 196, device = 'cpu', dropout = 0.1):
        super(RotaryPositionalEncoding, self).__init__()

        assert d_model % 2 == 0, "d_model must be even for RoPE"

        self.dropout = nn.Dropout(p = dropout)

        # Create a rotation matrix.
        self.rotation_matrix = torch.zeros(d_model, d_model, device = device)
        for i in range(d_model):
            for j in range(d_model):
                position = torch.tensor(i * j * 0.01)
                self.rotation_matrix[i, j] = torch.cos(position)

        # Create a positional embedding matrix.
        self.positional_embedding = torch.zeros(max_seq_len, d_model, device = device)
        for i in range(max_seq_len):
            for j in range(d_model):
                position = torch.tensor(i * j * 0.01)
                self.positional_embedding[i, j] = torch.cos(position)

    def forward(self, x):
        """ applies rotational encoding """
    
        try: x += self.positional_embedding
        except RuntimeError:

            # interpolate dimensions to be compatiable with x
            pos_emb_resized = F.interpolate(
                self.positional_embedding.unsqueeze(0).permute(0, 2, 1),  # [1, 512, 196]
                size = x.size(1),
                mode = "linear",
                align_corners = False
            ).permute(0, 2, 1)  # turn to [1, 784, 512]

            x += pos_emb_resized

        # Apply the rotation matrix to the input tensor.
        x = torch.matmul(x, self.rotation_matrix)

        return self.dropout(x)
    
class FiLM(nn.Module):
    def __init__(self, cond_dim: int):

        """ Affine Transformation For Input Features (text)
            Args: cond_dim: Embedding size For Both Image And Text (dimension)
        """
        # cond_dim === embed_dim

        super(FiLM, self).__init__() 
        self.film_layer = nn.Linear(cond_dim, cond_dim * 2)

    def forward(self, x, cond):
        # cond: conditioned text features 
        # return modulated features

        # applies film layer
        gamma_beta = self.film_layer(cond)
        # split gamma and beta
        gamma, beta = torch.chunk(gamma_beta, chunks = 2, dim = -1)

        # unsqueeze to allow broadcasting
        #return gamma.unsqueeze(1) * x + beta.unsqueeze(1)
        return gamma * x + beta
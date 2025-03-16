import torch
import torch.nn as nn
from torch import Tensor
from tokenizer import TorchTokenizer

def patchify(x: Tensor, size: int, stride: int):
    " Turn a latent into patches "
    # usually, the smaller the patch size the better quality of the generated image
    # if you have enough gpu power, try training with patch size of 2
    batch, channels, height, width = x.shape

    assert height % 4 == 0 and width % 4 == 0, "Image must be divisible into 16 patches"
    size = height // 4  # Ensuring 4x4 patches

    stride = size  # no overlap

    patches = x.unfold(2, size, stride).unfold(3, size, stride)

    # (batch, num_patches, channels, patch_size, patch_size)
    patches = patches.contiguous().view(batch, -1, channels, size, size)

    return patches

def depatchify(x: Tensor, img_size: int) -> torch.Tensor:
    " turn patches into an image "

    # check if image is 4 by 4 patches
    batch, patches, channels, size, _ = x.shape

    grid_size = int(patches * 0.5)
    assert grid_size * size == img_size, "Image must be 4x4 patches"

    # turning patches into images 
    x = x.view(batch, grid_size, grid_size, channels, size, size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(batch, channels, img_size, img_size) 

    return x
        
class ConditionalPromptNorm(nn.Module):
    # normalization with Feed-Forward layer for text prompts
    # should be encoded with clip
    def __init__(self, hidden_size, dim: int):
        super(ConditionalPromptNorm, self).__init__()

        # normalization
        # elementwise_affine will put biases into zeros and weights into ones
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine = False)
        # FF layer
        self.fcw = nn.Linear(dim, hidden_size)
        self.fcb = nn.Linear(dim, hidden_size)

    def forward(self, x: Tensor, features):   
        bs = x.size(0) # batch size

        out = self.norm(x)
        w = self.fcw(features).reshape(bs, 1, -1)
        b = self.fcb(features).reshape(bs, 1, -1)

        return  w * out + b

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, device = 'cpu', dropout = 0.1):
        super(RotaryPositionalEncoding, self).__init__()

        assert d_model % 2 == 0, "d_model must be even for RoPE"

        self.dropout = nn.Dropout(p = dropout)

        # Create a rotation matrix.
        self.rotation_matrix = torch.zeros(d_model, d_model, device = device)
        for i in range(d_model):
            for j in range(d_model):
                self.rotation_matrix[i, j] = torch.cos(i * j * 0.01)

        # Create a positional embedding matrix.
        self.positional_embedding = torch.zeros(max_seq_len, d_model, device = device)
        for i in range(max_seq_len):
            for j in range(d_model):
                self.positional_embedding[i, j] = torch.cos(i * j * 0.01)

    def forward(self, x):
        """ applies rotational encoding """

        # Add the positional embedding to the input tensor.
        x += self.positional_embedding

        # Apply the rotation matrix to the input tensor.
        x = torch.matmul(x, self.rotation_matrix)

        return self.dropout(x)
    
class HandlePrompt(nn.Module):
    # be used in training with raw string data
    def __init__(self):
        super(HandlePrompt, self).__init__()

        self.tokenizer = TorchTokenizer()
        self.norm = ConditionalPromptNorm()

    def forward(self, prompt: str):
        x = self.tokenizer.tokenize(prompt)
        x = self.norm(x)

        return x
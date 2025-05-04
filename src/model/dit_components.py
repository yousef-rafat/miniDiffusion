import math
import torch
import torch.nn as nn
from torch import Tensor
from clip import CLIP
#from t5_encoder import T5EncoderModel
from tokenizer import TorchTokenizer, UnigramTokenizer

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

class HandlePrompt(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip_tokenizer = TorchTokenizer()
        self.t5_tokenizer = UnigramTokenizer()
    
    def forward(self, x: torch.Tensor, clip: CLIP, t5_encoder): # return T5EncoderModel later

        clip_tokens = self.clip_tokenizer.tokenize(x)
        t5_tokens = self.t5_tokenizer.encode(x)

        clip_embeds = clip.encode_text(clip_tokens)
        t5_embeds = t5_encoder(t5_tokens)

        return clip_embeds, t5_embeds
    
class TimeStepEmbeddings(nn.Module):
    def __init__(self, in_features: int = 256, out_features: int = 2432):
        super().__init__()

        self.linear_1 = nn.Linear(in_features, out_features)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, out_features)

    def forward(self, x: Tensor) -> Tensor:

        x = self.linear_1(x)
        x = self.act(x)
        return self.linear_2(x)
    
class PixArtAlphaTextProjection(TimeStepEmbeddings):
    def __init__(self, in_features = 2048, out_features = 2432):
        super().__init__(in_features, out_features)

class AdaLayerNormContinuous(nn.Module):
    """ Modulate Image Features With Text Features """
    def __init__(self, embed_dim: int = 2432):
        super().__init__()

        self.silu = nn.SiLU()
        # linearly project it so we can split into scale and shift later
        self.linear = nn.Linear(embed_dim, embed_dim * 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, cond_embed):

        embed = self.silu(cond_embed)
        embed = self.linear(embed)
        # how to much to scale and shift each channel
        scale, shift = torch.chunk(embed, 2, dim = 1)

        # normalize input and multipled with scale and shift (after adding a new second dimension)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]

        return x
    
class AdaLayerNormZero(AdaLayerNormContinuous):
    def __init__(self, in_features = 2432, out_features = 14592):
        super().__init__(in_features, out_features)

    def forward(self, x):

        emb = self.linear(self.silu(x))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]

        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
    
class FeedForward(nn.Module):
    def __init__(self, dim: int = 2432, dropout_rate: float = 0.0):
        super().__init__()

        hidden_dim = dim * 4
        self.net = nn.ModuleList([
            nn.Sequential(
                nn.GELU(),
                nn.Linear(dim, hidden_dim)
            ),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim)
        ])


    def forward(self, x):
        
        for layer in self.net:
            x = layer(x)

        return x
    
class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 16, out_channels: int = 2432, kernel_size: int = 2, stride: int = 2):
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, bias = False)
    def forward(self, x):
        return self.proj(x)
    
class RMSNorm(nn.Module):
    """ RMSNorm Norm Implementation (different from nn.LayerNorm) """
    def __init__(self, hidden_size: int):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = 1e-6

    def forward(self, hidden_states):

        variance = hidden_states.pow(2).mean(-1, keepdim = True)    
        hidden_states *= torch.rsqrt(variance + self.eps) # 1 / sqrt(variance)

        return self.weight * hidden_states
    
def get_timestep_embeddings(timesteps: Tensor, embed_dim: int, max_period: int = 10_000, downscale_freq_shift: float = 1) -> Tensor:
    """ Get the sin-cos time embeddings """
    # time step embeddings for diffusion models are important as it gives
    # diffusion models a sense of time or progression and how much noise there is

    assert timesteps.dim() == 1, "timesteps dimensions must be equal to one"

    # split embeddings into two halves, one for sin and the other for cos
    half_dim = embed_dim // 2

    exponent = -math.log(max_period) * torch.arange(0, half_dim)

    # to spread the exponents evenly
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)

    # multiply the exponents with the timesteps
    emb = emb[:, None] * timesteps[None, :] # to do matrix mat

    #concat sin and cos embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim = -1)

    return emb

class Timesteps(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
    
    def forward(self, timesteps):

        time_embeddings = get_timestep_embeddings(
            timesteps = timesteps,
            embed_dim = self.embed_dim
        )

        return time_embeddings

import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from clip import CLIP, OpenCLIP
from tokenizer import TorchTokenizer, UnigramTokenizer

class HandlePrompt(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip_tokenizer = TorchTokenizer()
        self.t5_tokenizer = UnigramTokenizer()
    
    def forward(self, x: str, clip: CLIP, clip_2: OpenCLIP, t5_encoder):

        clip_tokens = self.clip_tokenizer.tokenize(x)
        t5_tokens = self.t5_tokenizer.encode(x)

        pooled, clip_embeds = clip.encode_text(clip_tokens)
        pooled2, clip_2_embeds = clip_2.encode_text(clip_tokens)

        t5_embeds = t5_encoder(t5_tokens)

        clip_embeddings = torch.cat([clip_embeds, clip_2_embeds], dim = -1)

        # get the difference between t5 and clip embeddings and pad it for it to become a matrix
        clip_embeddings = F.pad(
            clip_embeddings, (0, t5_embeds.size(-1) - clip_embeddings.size(-1))
        )

        embeddings = torch.cat([clip_embeddings, t5_embeds], dim = -2)
        pooled_embeddings = torch.cat([pooled, pooled2], dim = -1)

        return embeddings, pooled_embeddings
    
class TimeStepEmbeddings(nn.Module):
    def __init__(self, in_features: int = 256, out_features: int = 1536):
        super().__init__()

        self.linear_1 = nn.Linear(in_features, out_features)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, out_features)

    def forward(self, x: Tensor) -> Tensor:

        x = self.linear_1(x)
        x = self.act(x)
        return self.linear_2(x)
    
class PixArtAlphaTextProjection(TimeStepEmbeddings):
    def __init__(self, in_features = 1536, out_features = 2048):
        super().__init__()
        
        self.linear_1 = nn.Linear(out_features, in_features)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(in_features, in_features)

class AdaLayerNormContinuous(nn.Module):
    """ Modulate Image Features With Text Features """
    def __init__(self, embed_dim: int = 2432):
        super().__init__()

        self.silu = nn.SiLU()
        # linearly project it so we can split into scale and shift later
        self.linear = nn.Linear(embed_dim, embed_dim * 2)
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine = False)

    def forward(self, x, cond_embed):

        embed = self.silu(cond_embed)
        embed = self.linear(embed)
        # how to much to scale and shift each channel
        scale, shift = torch.chunk(embed, 2, dim = 1)

        # normalize input and multiply with scale and shift (after adding a new second dimension)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]

        return x
    
class AdaLayerNormZero(AdaLayerNormContinuous):
    def __init__(self, in_features = 2432):
        super().__init__(in_features)
        self.linear = nn.Linear(in_features, in_features * 6)

    def forward(self, x: Tensor, embed: Tensor):

        emb = self.linear(self.silu(embed))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        #                   multiplicative scale      bias
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]

        # (hidden_states, scale msa output before residual, bias before mlp, multiplicative scale before mlp)
        # gate_mlp = gate for scaling before residual connection
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
    
class AdaLayerNormZeroX(AdaLayerNormContinuous):
    def __init__(self, embed_dim = 1536):
        super().__init__(embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim * 9)

    def forward(self, hidden_states: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:

        embed = self.linear(self.silu(embed))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = embed.chunk(
            9, dim = 1
        )
        norm_hidden_states = self.norm(hidden_states)

        # multiply by weights and add bias to the normalized hidden states
        hidden_states = norm_hidden_states * (1 + scale_msa[:, None]) + shift_msa[:, None]
        norm_hidden_states2 = norm_hidden_states * (1 + scale_msa2[:, None]) + shift_msa2[:, None]

        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2

    
class GELU(nn.Module):
    " GELU with tanh approximation and a linear layer "
    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim_out)

    def gelu(self, x):
        return F.gelu(x, approximate = "tanh")
    
    def forward(self, x):
        x = self.proj(x)
        x = self.gelu(x)

        return x
    
class FeedForward(nn.Module):
    def __init__(self, dim: int = 1536, dropout_rate: float = 0.0):
        super().__init__()

        hidden_dim = dim * 4
        self.net = nn.ModuleList([
            GELU(dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim),
        ])

    def forward(self, x):
        
        for layer in self.net:
            x = layer(x)

        return x
    
class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 16, embedding_dim: int = 2432, image_size: int = 384, patch_size: int = 16,
                 pos_embed_max_size: int = 384):
        super().__init__()

        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size = 2, stride = 2)
        self.pos_embed_max_size = pos_embed_max_size

        self.height = self.width = image_size // patch_size
        self.base_size = self.height // patch_size

        # precompute positional embeddings patches of images
        pos_embed = self.get_2d_sincos_pos_embed(
            embed_dim = embedding_dim,
            grid_size = self.pos_embed_max_size,
            base_size = self.base_size
        )

        self.patch_size = patch_size
        self.size = image_size

        self.register_buffer("pos_embed", pos_embed.unsqueeze(0).float(), persistent = True)

    def forward(self, x):
        
        height, width = x.shape[-2:]
        x = self.proj(x)
        x = x.flatten(2).transpose(2, 1)
    
        pos_embed = self.cropped_pos_embed(height, width)

        return (x + pos_embed)
    
    def cropped_pos_embed(self, height: int, width: int):

        # extract centered crop from positional embeddinging
        # this is what SD3.5 does

        height //= self.patch_size
        width //= self.patch_size

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2

        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top: top + height, left: left + width, :]

        return spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])

    
    def get_2d_sincos_pos_embed(self, embed_dim: int, grid_size: int, base_size = 16):

        # compute sincos embeddings for all the possible grids

        grid_h = torch.arange(grid_size) / (grid_size / base_size)
        grid_w = torch.arange(grid_size) / (grid_size / base_size)

        # create indexing (0,0), (0,1),... for each position of a patch
        # returns the x coordinates and y coordinates for each no. of patches and stack them together
        grid = torch.meshgrid(grid_w, grid_h, indexing = "xy")
        grid = torch.stack(grid, dim = 0)

        # ensure correct shape
        grid = grid.reshape(2, 1, grid_size, grid_size)

        # actual function that gets the positional embeddings
        return PatchEmbed.get_2d_sincos_pod_embed_grid(embed_dim, grid)

    @staticmethod
    def get_2d_sincos_pod_embed_grid(embed_dim, grid):
        # get 1d sincos embeddings and combine them into 2d
        emb_h = PatchEmbed.get_1d_sincos_pos_embed_grid(embed_dim // 2, grid[0])
        emb_w = PatchEmbed.get_1d_sincos_pos_embed_grid(embed_dim // 2, grid[1])
        return torch.cat([emb_h, emb_w], dim=1)
    
    @staticmethod
    def get_1d_sincos_pos_embed_grid(embed_dim, pos):

        # generate vector of frequencies
        omega = torch.arange(embed_dim // 2)
        omega = omega.float() / (embed_dim / 2.0)
        omega = 1.0 / 10000 ** omega

        pos = pos.reshape(-1)
        out = torch.outer(pos, omega)

        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    
class RMSNorm(nn.Module):
    """ RMSNorm Norm Implementation (different from nn.LayerNorm) """
    def __init__(self, hidden_size: int):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = 1e-6

    def forward(self, hidden_states):

        variance = hidden_states.pow(2).mean(-1, keepdim = True)    
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps) # 1 / sqrt(variance)

        return self.weight * hidden_states
    
def get_timestep_embeddings(timesteps: Tensor, embed_dim: int = 256, max_period: int = 10_000) -> Tensor:
    """ Get the sin-cos time embeddings """
    # time step embeddings for diffusion models are important as it gives
    # diffusion models a sense of time or progression and how much noise there is

    assert len(timesteps.shape) == 1, "timesteps dimensions must be equal to one"

    # split embeddings into two halves, one for sin and the other for cos
    half_dim = embed_dim // 2

    exponent = -math.log(max_period) * torch.arange(start = 0, end = half_dim)

    # to spread the exponents evenly
    exponent = exponent / half_dim

    emb = torch.exp(exponent)

    # multiply the exponents with the timesteps
    emb = timesteps[:, None].float() * emb[None, :]  # to do matrix mat

    #concat sin and cos embeddings
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim = -1)

    return emb

class Timesteps(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, timesteps):

        time_embeddings = get_timestep_embeddings(
            timesteps = timesteps,
            embed_dim = 256
        )

        return time_embeddings

class CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_proj = Timesteps()
        self.timestep_embedder = TimeStepEmbeddings()
        self.text_embedder = PixArtAlphaTextProjection()

    def forward(self, timestep, pooled_projection: Tensor) -> Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)

        pooled_projections = self.text_embedder(pooled_projection)

        conditioning = timesteps_emb + pooled_projections

        return conditioning
    
def chunk_feed_forward(ff: FeedForward, inputs: Tensor, chunk_dim: int, chunk_size: int):
    
    num_chunks = inputs[chunk_dim] // chunk_size
    outputs = torch.cat([ff(chunk) for chunk in inputs.chunk(num_chunks, dim = chunk_dim)],
                        dim = chunk_dim)

    return outputs
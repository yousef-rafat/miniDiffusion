import math
import torch
import torch.nn as nn
from torch import Tensor
from clip import CLIP
from t5_encoder import T5EncoderModel
from tokenizer import TorchTokenizer, UnigramTokenizer

class HandlePrompt(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip_tokenizer = TorchTokenizer()
        self.t5_tokenizer = UnigramTokenizer()
    
    def forward(self, x: torch.Tensor, clip: CLIP, t5_encoder: T5EncoderModel):

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

        # normalize input and multiply with scale and shift (after adding a new second dimension)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]

        return x
    
class AdaLayerNormZero(AdaLayerNormContinuous):
    def __init__(self, in_features = 2432):
        out_features = in_features * 6
        super().__init__(in_features, out_features)

    def forward(self, x):

        emb = self.linear(self.silu(x))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        #                   multiplicative scale      bias
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]

        # (hidden_states, scale msa output before residual, bias before mlp, multiplicative scale before mlp)
        # gate_mlp = gate for scaling before residual connection
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
    def __init__(self, in_channels: int = 16, embedding_dim: int = 2432, image_size: int = 256, patch_size: int = 16):

        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size = 2, stride = 2, bias = False)
        self.pos_embed_max_size = image_size // patch_size

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
        
        x = self.proj(x)

        pos_embed = self.cropped_pos_embed(self.size, self.size)

        return (x + pos_embed)
    
    def cropped_pos_embed(self, height: int, width: int):

        # extract centered crop from positional embeddinging
        # this is what SD3.5 does

        height //= self.patch_size
        width //= self.patch_size

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2

        spatial_pos_embed = self.pos_emed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top: top + height, left: left + width, :]

        return spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])

    
    def get_2d_sincos_pos_embed(self, embed_dim: int, grid_size: int, base_size = 16):

        # compute sincos embeddings for all the possible grids

        grid_h = torch.arange(grid_size) / (grid_size / base_size)
        grid_w = torch.arange(grid_size) / (grid_size / base_size)

        grid = torch.meshgrid(grid_w, grid_h, indexing = "xy")
        grid = torch.stack(grid, dim = 0).reshape(2, 1, grid_size, grid_size)

        return PatchEmbed.get_2d_sincos_pod_embed_grid(embed_dim, grid)

    @staticmethod
    def get_2d_sincos_pod_embed_grid(embed_dim, grid_size):
        emb_h = PatchEmbed.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_size)
        emb_w = PatchEmbed.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_size)
        return torch.cat([emb_h, emb_w], dim=1)
    
    @staticmethod
    def get_1d_sincos_pos_embed_grid(embed_dim):

        omega = torch.arange(embed_dim // 2)
        omega /= embed_dim / 2.0
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
        hidden_states *= torch.rsqrt(variance + self.eps) # 1 / sqrt(variance)

        return self.weight * hidden_states
    
def get_timestep_embeddings(timesteps: Tensor, embed_dim: int, max_period: int = 10_000, downscale_freq_shift: float = 1,
                            flip_sin_cos: bool = True) -> Tensor:
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

    # flip to (cos, sin) if set to True
    if flip_sin_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim = -1)

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

class CombinedTimestepTextProjEmbeddings(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.time_proj = Timesteps(embed_dim)
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
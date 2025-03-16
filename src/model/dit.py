import torch
import torch.nn as nn
from attention import PagedTransformerEncoderLayer
from dit_components import patchify, depatchify, RotaryPositionalEncoding

class DiTBlock(nn.Module):
    # reference for rectified flow: https://arxiv.org/pdf/2403.03206
    def __init__(self, heads: int, embedding_size: int = 512, patch_size: int = 16, depth: int = 5, max_length: int = 1024):
        super(DiTBlock, self).__init__()

        #####################
        # Create model layers
        #####################

        # module list so we can pass the attention mask for every layer
        self.model = nn.ModuleList(
            [PagedTransformerEncoderLayer(heads, embedding_size) for _ in range(depth)]
        )

        self.p_size = patch_size
        vocab_size = 50261

        self.token_embedding = nn.Embedding(vocab_size, embedding_size)

        self.time_mlp = nn.Sequential([
            RotaryPositionalEncoding(embedding_size, max_length),
            nn.Linear(embedding_size, embedding_size * 2),
            nn.GELU(),
            nn.Linear(embedding_size * 2, embedding_size),
            nn.GELU()
        ])

        self.linear_patches = nn.Linear(patch_size, embedding_size)
    
    def forward(self, latent: torch.Tensor, t: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

        " Goal: Given Space, Time, and Conditioned Prompt, Return Velocity Vector "

        # t = current_time [0, 1]

        patches = patchify(latent, size = self.p_size)
        patches = self.linear_patches(patches)

        # create time based embeddings
        time_embeddings = self.time_mlp(patches)

        # (batch, seq_len, embed_dim)
        tokens = self.token_embedding(input_ids)

        x = torch.cat([tokens, patches], dim = 1)

        # create zeros so time embeddings won't interfere with text (tokens) embeddings
        zeros_text = torch.zeros(latent.size(0), tokens.size(1), time_embeddings.size(-1), device = time_embeddings.device)
        time_embeddings = torch.cat([zeros_text, time_embeddings])

        # make sure t has the correct shape
        # (batch_size, seq_len + num_patches, 1)
        t = t.unsqueeze(1).expand(-1, x.size(1), -1) # broadcast

        x = x + time_embeddings + t

        # adjust attention mask to handle image patches
        # (batch_size, num_patches + seq_len)
        full_attention_mask = torch.cat([attention_mask, torch.ones(input_ids.size(0), patches.size(1), device = attention_mask.device)])

        src_key_padding_mask = ~full_attention_mask.bool()

        # pass attention mask for every layer
        for layer in self.model:
            x = layer(x, src_key_padding_mask = src_key_padding_mask)

        # remove text processing from image
        x = x[:, attention_mask.size(1):, :]

        output = depatchify(x, img_size = latent.size(2))

        return output
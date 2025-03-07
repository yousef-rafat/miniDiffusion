import torch
import torch.nn as nn
from attention import PagedTransformerEncoderLayer
from dit_components import patchify, depatchify, RotaryPositionalEncoding

class DiTBlock(nn.Module):
    # reference for rectified flow: https://arxiv.org/pdf/2403.03206
    def __init__(self, heads: int, embedding_size: int, size: int = 16, depth: int = 5):
        super(DiTBlock, self).__init__()

        #####################
        # Create model layers
        #####################

        # module list so we can pass the attention mask for every layer
        self.model = nn.ModuleList(
            [PagedTransformerEncoderLayer(heads, embedding_size) for _ in range(depth)]
        )

        self.size = size
        vocab_size = 50261

        self.token_embedding = nn.Embedding(vocab_size, embedding_size)

        self.time_mlp = nn.Sequential([
            RotaryPositionalEncoding(embedding_size, 1024),
            nn.Linear(embedding_size, embedding_size * 2),
            nn.GELU(),
            nn.Linear(embedding_size * 2, embedding_size),
            nn.GELU()
        ])
    
    def forward(self, latent: torch.Tensor, t: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # t = current_time [0, 1]

        patches = patchify(latent, size = self.size)

        # create time based embeddings
        time_embeddings = self.time_mlp(patches)

        # (batch, seq_len, embed_dim)
        tokens = self.token_embedding(input_ids)

        x = torch.cat([tokens, patches], dim = 1)

        # create zeros so time embeddings won't interfere with text (tokens) embeddings
        zeros_text = torch.zeros(latent.size(0), tokens.size(1), time_embeddings.size(-1), device = time_embeddings.device)
        time_embeddings = torch.cat([zeros_text, time_embeddings])

        # make sure t has the correct shape
        t = t.expand(x.size(0), 1)
        x = torch.cat([x, t], dim = 1)

        x = time_embeddings + x

        src_key_padding_mask = ~attention_mask.bool()

        # pass attention mask for every layer
        for layer in self.model:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        # remove text processing from image
        x = x[:, attention_mask.size(1):, :]

        output = depatchify(x, img_size = latent.size(2))

        return output
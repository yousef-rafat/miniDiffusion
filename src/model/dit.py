import torch
import torch.nn as nn
from model.attention import PagedTransformerEncoderLayer
from model.dit_components import patchify, depatchify, RotaryPositionalEncoding, FiLM

# MultiModal Diffiusion Transformer Block (MMDiT)
class DiT(nn.Module):
    # reference for rectified flow: https://arxiv.org/pdf/2403.03206
    def __init__(self, heads: int = 8, embedding_size: int = 512, patch_size: int = 8, depth: int = 5, max_length: int = 196):
        super(DiT, self).__init__()

        #####################
        # Create model layers
        #####################

        # module list so we can pass the attention mask for every layer
        self.model = nn.ModuleList(
            [PagedTransformerEncoderLayer(num_heads = heads, embed_dim =  embedding_size) for _ in range(depth)]
        )

        self.embed_size = embedding_size
        self.p_size = patch_size

        self.film = FiLM(embedding_size)

        self.time_mlp = nn.Sequential(
            RotaryPositionalEncoding(embedding_size, max_seq_len = max_length),
            nn.Linear(embedding_size, embedding_size * 2),
            nn.GELU(),
            nn.Linear(embedding_size * 2, embedding_size),
            nn.GELU()
        )

        self.linear_patches = nn.Linear(patch_size * patch_size * 4, embedding_size)

    def handle_patches(self, latent: torch.Tensor):

        """ Creates and returns patches and time embeddings """

        patches = patchify(latent, size = self.p_size)

        # make matmul be patch size by patch size (ij,jk -> ik)
        B, N, C, H, W = patches.shape
        patches = self.linear_patches(patches.view(B, N, C * H * W)) # flatten dimensions

        # create time based embeddings
        time_embeddings = self.time_mlp(patches)

        return patches, time_embeddings
    
    def forward(self, latent: torch.Tensor, t: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

        " Goal: Given Space, Time, and Conditioned Prompt, Return Velocity Vector "

        # t = current_time [0, 1]

        patches, time_embeddings = self.handle_patches(latent)

        # concatenate patches and tokens
        # resize for torch.cat
        patches = patches.view(patches.size(0), -1, self.embed_size)
        x = torch.cat([input_ids, patches], dim = 1)

        # create zeros so time embeddings won't interfere with text (tokens) embeddings
        zeros_text = torch.zeros(latent.size(0), input_ids.size(1), time_embeddings.size(-1), device = time_embeddings.device)
        time_embeddings = torch.cat([zeros_text, time_embeddings], dim = 1)

        # adjust attention mask to handle image patches
        # (batch_size, num_patches + seq_len)
        full_attention_mask = torch.cat([attention_mask, torch.ones(input_ids.size(0), patches.size(1), device = attention_mask.device)], dim = 1)

        src_key_padding_mask = ~full_attention_mask.bool()

        # make sure t has the correct shape
        # (batch_size, seq_len + num_patches, 1)
        t = t.squeeze(-1).expand(-1, x.size(1), -1) # broadcast

        x = (self.film(x, cond = time_embeddings) + t).squeeze(0)

        # pass attention mask for every layer
        for layer in self.model:
            x = layer(x, src_key_padding_mask = src_key_padding_mask)

        # remove text processing from image
        x = x[:, attention_mask.size(1):, :].reshape(x.size(0), -1 , 4, self.p_size, self.p_size)

        output = depatchify(x, img_size = 40)

        return output
    
    def solve(self, latent: torch.Tensor, t: torch.Tensor):

        """ Solve ODE in reversing the flow of the velocity vector
            Returns the denoised latent
        """

        x, time_embeddings = self.handle_patches(latent)

        t = t.view(-1, 1, 1)
        t = t.expand(x.size(0), x.size(1), -1) 

        x = (self.film(x, cond = time_embeddings) + t).squeeze(0)

        for layer in self.model:
            x = layer(x)

        x = x.reshape(x.size(0), -1 , 4, self.p_size, self.p_size)

        output = depatchify(x, img_size = 40)

        return output

def test_dit():

    dit = DiT()
    latent = torch.randn(1, 4, 28, 28)

    input_ids = torch.randint(50000, size = (1, 1024))
    attention_mask = torch.zeros(input_ids.size(1)).unsqueeze(0)

    t = torch.rand(1, 1) # timestep for each batch

    output = dit(latent = latent, input_ids = input_ids, attention_mask = attention_mask, t = t)

    print(output)
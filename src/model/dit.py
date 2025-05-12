import torch
import torch.nn as nn
from model.attention import PagedJointAttention
from model.dit_components import ( 
    CombinedTimestepTextProjEmbeddings, PatchEmbed, AdaLayerNormContinuous, AdaLayerNormZero,
    FeedForward, chunk_feed_forward
)

####################################################################################
# SD3 Architecture: https://learnopencv.com/wp-content/uploads/2024/11/SD35_arch.png
####################################################################################

class DiTBlock(nn.Module):
    def __init__(self, num_heads: int, embedding_size: int = 1536, add_context_final: bool = False, dropout_rate: float = 0.0):
        super().__init__()

        """
        add_context_final: add special processings for the final layer in the model
        """

        self.norm_1 = AdaLayerNormZero(in_features= embedding_size)

        self.final_context = add_context_final

        if add_context_final:
            self.norm1_context = AdaLayerNormContinuous(embed_dim = embedding_size)
        else: self.norm1_context = AdaLayerNormZero(in_features = embedding_size)

        self.attn = PagedJointAttention(embedding_size = embedding_size, num_heads = num_heads, dropout = dropout_rate)
        self.ff = FeedForward(dropout_rate = dropout_rate)

        # for ff layer
        self.norm2 = nn.LayerNorm(embedding_size)

        if add_context_final:
            self.ff_context = FeedForward(dim = embedding_size)
            self.norm2_context = nn.LayerNorm(embedding_size, elementwise_affine = False)
        else:
            self.ff_context = None
            self.norm2_context = None

        # TODO: figure out correct or optimized values
        self.chunk_dim = 1
        self.chunk_size = 256

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, time_emb: torch.Tensor, use_chunking: bool = True):

        # /////////
        # Normalize
        # /////////

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb = time_emb)
        
        if self.final_context:
            norm_hidden_states = self.norm1_context(encoder_hidden_states)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb = time_emb
            )

        ##############################################################################################################
        # Attention
        attn_output, encoder_output = self.attn(norm_hidden_states, encoder_hidden_state = norm_encoder_hidden_states)

        # gated residual attention connection
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states += attn_output

        #########################################################################################################################
        # Feed Forward
        norm_hidden_states = self.norm2(hidden_states)
        # scale and shift hidden states
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if use_chunking:
            ff_output = chunk_feed_forward(self.ff, norm_hidden_states, chunk_dim = self.chunk_dim, chunk_size = self.chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states += ff_output

        #############################################################################################################################
        # Fead Forward Context
        if self.final_context:
            encoder_hidden_states = None

        else:
            # apply attention mechanisim
            encoder_output = c_gate_msa.unsqueeze(1) * encoder_output
            encoder_hidden_states += encoder_output

            # normalize and (shift and scale)
            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

            # apply ff context
            if use_chunking:
                context_ff_output = chunk_feed_forward(self.ff_context, norm_encoder_hidden_states, self.chunk_dim, self.chunk_size)
            else: context_ff_output = self.ff_context(norm_encoder_hidden_states)

            # apply gated residual connection
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, encoder_hidden_states

# MultiModal Diffiusion Transformer Block (MMDiT)
class DiT(nn.Module):
    # reference for rectified flow: https://arxiv.org/pdf/2403.03206
    def __init__(self, heads: int = 8, embedding_size: int = 1536, patch_size: int = 16, depth: int = 5, caption_projection_dim: int = 1152,
                 output_channels: int = 16):
        super(DiT, self).__init__()

        #####################
        # Create model layers
        #####################

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(embed_dim = embedding_size)
        self.pos_embed = PatchEmbed(embedding_dim = embedding_size, patch_size = patch_size)

        self.norm_out = AdaLayerNormContinuous(embed_dim = embedding_size)
        self.context_embedder = nn.Linear(embedding_size, caption_projection_dim)

        self.embed_size = embedding_size
        self.patch_size = patch_size
        self.output_channels = output_channels

        self.transformer_blocks = nn.ModuleList([

            DiTBlock(
                num_heads = heads,
                embedding_size = embedding_size,
                add_context_final = i == (depth - 1)
            )

            for i in range(depth)
            ])

        self.proj_out = nn.Linear(embedding_size, patch_size * patch_size * output_channels)
    
    def forward(self, latent: torch.Tensor, encoder_hidden_states: torch.Tensor, timestep: torch.LongTensor,
                pooled_projections: torch.Tensor) -> torch.Tensor:

        """ Goal: Given Space, Time, and Conditioned Prompt, Return Velocity Vector 
            Input: Noised Latent, Specific Timestep, Pooled Text Projections, Encoded Text Embeddings
            Output: Velocity vector
        """

        height, width = latent.shape[-2:]
        hidden_states = self.pos_embed(latent)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        time_embeddings = self.time_text_embed(timestep, pooled_projections)

        # pass attention mask for every layer
        for layer in self.transformer_blocks:

            hidden_states = layer(
                hidden_states = hidden_states,
                encoder_hidden_states = encoder_hidden_states,
                time_embeddings = time_embeddings,
                use_chunking = True
            )

        hidden_states = self.norm_out(hidden_states, time_embeddings)
        hidden_states = self.proj_out(hidden_states)

        ##########################################################################################################
        # Depatchify

        height //= self.patch_size
        width //= self.patch_size

        hidden_states.reshape(
            shape = (hidden_states.size(0), height, width, self.patch_size, self.patch_size, self.output_channels)
        )

        # permute tensor using einsum
        # original = (batch, height, width, patch_size, patch_size, channels)
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)

        output = hidden_states.reshape(
            shape=(hidden_states.size(0), self.output_channels, height * self.patch_size, width * self.patch_size)
        )
        
        return output

def test_dit():
    # TODO: fix the DiT testing
    dit = DiT()
    latent = torch.randn(1, 4, 28, 28)

    input_ids = torch.randint(50000, size = (1, 1024))
    attention_mask = torch.zeros(input_ids.size(1)).unsqueeze(0)

    t = torch.rand(1, 1) # timestep for each batch

    output = dit(latent = latent, input_ids = input_ids, attention_mask = attention_mask, t = t)

    print(output)

if __name__ == "__main__":
    test_dit()
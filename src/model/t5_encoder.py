############################################################################## REIMPLEMENTATION OF T5 Encoder ############################################################################

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.dit_components import RMSNorm
    
class DenseRelu(nn.Module):
    """ T5 feed-forward with gating: wo(GELU(wi_0(x)) * wi_1(x)) """
    def __init__(self, embed_size: int, ff_size: int):
        super().__init__()

        # ff_size = fead-forward size (dimensions)

        self.wi_1 = nn.Linear(embed_size, ff_size, bias = False)
        self.wi_0 = nn.Linear(embed_size, ff_size, bias = False)
        self.wo = nn.Linear(ff_size, embed_size, bias = False)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.wo(self.activation(self.wi_0(x)) * self.wi_1(x))

class T5LayerFF(nn.Module):
    def __init__(self, embed_dim: int, ff_size: int):
        super().__init__()

        self.layer_norm = RMSNorm(embed_dim)
        self.DenseReluDense = DenseRelu(embed_size = embed_dim, ff_size = ff_size)

    def forward(self, hidden_states):
        
        # forward pass
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)

        # residual connection
        hidden_states = hidden_states + forwarded_states

        return hidden_states
class SelfAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, index: int, relative_attention_bias: int = 32, max_length: int = 128):
        super().__init__()

        assert embed_size % num_heads == 0, "embedding size must be divisable by no. of heads"

        # query, key, value, output proj
        self.q = nn.Linear(embed_size, embed_size, bias = False)
        self.k = nn.Linear(embed_size, embed_size, bias = False)
        self.v = nn.Linear(embed_size, embed_size, bias = False)
        self.o = nn.Linear(embed_size, embed_size, bias = False)

        self.head_dim = embed_size // num_heads
        self.n_heads = num_heads
        self.embed_size = embed_size

        self.first_index = index == 0

        # relative position (t5 specific)
        if self.first_index:
            self.relative_attention_bias = nn.Embedding(relative_attention_bias, num_heads)

        self.max_length = max_length
        self.num_buckets = relative_attention_bias

    def _relative_position_bucket(self, relative_position):
        """Copied from HuggingFace T5Attention._relative_position_bucket (slightly modified)"""
        num_buckets = self.num_buckets
        max_distance = self.max_length
        # if bidirectional, half for future, half for past
        relative_buckets = 0

        # encoder = bidirectional
        num_buckets //= 2
        # positions > 0 go to second half
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)

        # now relative_position is in [0, ∞)
        max_exact = num_buckets // 2
        is_small   = relative_position < max_exact

        # large positions get log-scaled buckets
        relative_if_large = max_exact + (

            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)

        ).to(torch.long)

        relative_if_large = torch.min(
            relative_if_large,
            torch.full_like(relative_if_large, num_buckets - 1)
        )

        return relative_buckets + torch.where(is_small, relative_position, relative_if_large)

    def compute_bias(self, seq_len, device):
        """Compute the (1, heads, seq_len, seq_len) bias tensor."""

        context_position = torch.arange(seq_len, device=device)[:, None]
        memory_position  = torch.arange(seq_len, device=device)[None, :]

        relative_position = memory_position - context_position  # (seq, seq)
        rp_bucket = self._relative_position_bucket(relative_position)

        values    = self.relative_attention_bias(rp_bucket)     # (seq, seq, heads)
        return values.permute(2, 0, 1).unsqueeze(0)             # (1, heads, seq, seq)
    
    def forward(self, x, position_bias = None, key_pad_mask = None):

        B, seq_len, _ = x.size()

        # reshape for multi-head attention
        query = self.q(x).view(B, -1, self.n_heads, self.head_dim).transpose(1,2)
        key = self.k(x).view(B, -1, self.n_heads, self.head_dim).transpose(1,2)
        value = self.v(x).view(B, -1, self.n_heads, self.head_dim).transpose(1,2)

        scores = torch.matmul(query, key.transpose(3, 2))
        key_len = key.size(-2)

        # relative-pos
        if position_bias is None:

            if self.first_index:
                position_bias = self.compute_bias(seq_len, x.device)
                # get from end till seq_len
                position_bias = position_bias[:, :, -seq_len:, :]
            else:
                # initalize position_bias to zeros
                position_bias = torch.zeros(
                    1, self.n_heads, seq_len, key_len, dtype = scores.dtype, device = scores.device
                )

            if key_pad_mask is not None:
                key_pad_mask = key_pad_mask[:, :, :, :key_len]
                position_bias = position_bias + key_pad_mask

        scores += position_bias

        attentions = F.softmax(scores, dim = -1)

        out = torch.matmul(attentions, value)
        out = out.transpose(1,2).contiguous().view(B, seq_len, self.embed_size)
        outputs = self.o(out)

        return outputs, position_bias

class T5LayerSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, index: int):
        super().__init__()

        self.SelfAttention = SelfAttention(embed_size = embed_dim, num_heads = num_heads, index = index)
        self.layer_norm = RMSNorm(hidden_size = embed_dim)

    def forward(self, hidden_states, pad_mask, position_bias = None):
        
        norm_hidden_states = self.layer_norm(hidden_states)
        attn_outputs, position_bias = self.SelfAttention(norm_hidden_states, key_pad_mask = pad_mask, position_bias = position_bias)

        # residual connection
        hidden_states = hidden_states + attn_outputs

        return hidden_states, position_bias
    
class T5Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, index):
        super().__init__()
        self.layer = nn.ModuleList([T5LayerSelfAttention(embed_dim, num_heads, index = index), T5LayerFF(embed_dim, ff_dim)])

    def forward(self, x, key_pad_mask = None, position_bias = None):

        hidden_states, position_bias = self.layer[0](x,
                                                     pad_mask = key_pad_mask,
                                                     position_bias = position_bias)
        x = self.layer[1](hidden_states)
        
        return x, position_bias

class T5EncoderModel(nn.Module):
    def __init__(self, embed_dim: int = 4096, num_heads: int = 64, ff_dim: int = 10240, depth: int = 24):
        super().__init__()
        vocab_size = 32128

        self.encoder = nn.Module()
        self.encoder.block = nn.ModuleList([T5Encoder(embed_dim, ff_dim = ff_dim, num_heads = num_heads, index = i) for i in range(depth)])
        self.encoder.final_layer_norm = RMSNorm(embed_dim)

        self.shared = nn.Embedding(vocab_size, embed_dim)
        self.encoder.embed_tokens = self.shared

    def forward(self, text, attn_mask: torch.Tensor):

        x = self.encoder.embed_tokens(text) 

        src_key_mask = attn_mask == 0 # invert attn_mask

        # expand the attention mask
        mask = src_key_mask[:, None, None, :].to(x.dtype)
        mask = mask * torch.finfo(x.dtype).min # min = -max
        
        # we create the position_bias and store them in the first layer,
        # then we share the position_bias with all of the rest of layers
        position_bias = None
        for block in self.encoder.block:
            x, position_bias = block(x, key_pad_mask = mask, position_bias = position_bias)

        x = self.encoder.final_layer_norm(x)

        return x

def load_t5(model: T5EncoderModel, device: str = "cpu") -> T5EncoderModel:

    checkpoint = os.path.join(os.getcwd(), "encoders", "hub", "checkpoints", "t5_encoder.pth")
    missing, unexpected = model.load_state_dict(torch.load(checkpoint), strict = True)
    model = model.to(device)

    # for debuggging
    if len(missing) != 0:
        print(f"Missing keys ({len(missing)}):", missing)
        print(f"\nUnexpected keys ({len(unexpected)}):", unexpected)

    model.eval()

    return model

def test_t5():

    from tokenizer import UnigramTokenizer

    tokenizer = UnigramTokenizer()

    tokens = tokenizer.encode_ids("a photo of a cat")
    tokens2 = tokenizer.encode_ids("a photo of a table")

    attn = torch.ones_like(tokens)
    attn2 = torch.ones_like(tokens2)

    model = T5EncoderModel()
    model = load_t5(model)

    import torch.nn.functional as F

    outputs = model(tokens.unsqueeze(0), attn.unsqueeze(0)).mean(dim = 1)
    outputs2 = model(tokens2.unsqueeze(0), attn2.unsqueeze(0)).mean(dim = 1)
    
    sim = F.cosine_similarity(outputs, outputs2)

    print(sim)

if __name__ == "__main__":
    test_t5()
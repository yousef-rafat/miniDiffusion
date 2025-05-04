############################################################################## REIMPLEMENTATION OF T5 Encoder ############################################################################

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dit_components import RMSNorm
    
class DenseRelu(nn.Module):
    """ T5 feed-forward with gating: wo(GELU(wi_0(x)) * wi_1(x)) """
    def __init__(self, embed_size: int, ff_size: int, drop_rate: float = 0.1):
        super().__init__()

        # ff_size = fead-forward size (dimensions)

        self.wi_1 = nn.Linear(embed_size, ff_size, bias = False)
        self.wi_0 = nn.Linear(embed_size, ff_size, bias = False)
        self.wo = nn.Linear(ff_size, embed_size, bias = False)
        self.activation = nn.GELU()

        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        return self.wo(self.dropout(self.activation(self.wi_0(x)) * self.wi_1(x)))

class SelfAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, index: int, relative_attention_bias: int = 32, max_length: int = 128, dropout_rate: float = 0.1):
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

        self.attn_dropout = nn.Dropout(dropout_rate)

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

    def forward(self, x, key_pad_mask = None):

        B, seq_len, _ = x.size()

        # reshape for multi-head attention
        query = self.q(x).view(B, seq_len, self.n_heads, self.head_dim).transpose(1,2)
        key = self.k(x).view(B, seq_len, self.n_heads, self.head_dim).transpose(1,2)
        value = self.v(x).view(B, seq_len, self.n_heads, self.head_dim).transpose(1,2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # relative-pos
        if self.first_index:
            bias = self.compute_bias(seq_len, x.device)
            scores += bias

        if key_pad_mask is not None:
            key_pad_mask = key_pad_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_pad_mask, float("-inf"))

        attentions = F.softmax(scores, dim = -1)

        attentions = self.attn_dropout(attentions)

        out = torch.matmul(attentions, value)
        out = out.transpose(1,2).contiguous().view(B, seq_len, self.embed_size)
        outputs = self.o(out)

        return outputs

class T5EncoderLayer(nn.Module):
    def __init__(self, embed_dim, ff_dim, index, num_heads: int = 12, attention: bool = False, dropout_rate: float = 0.1):
        super().__init__()

        self.attention = attention

        if attention:
            self.SelfAttention = SelfAttention(embed_size = embed_dim, num_heads = num_heads, index = index)
        else:
            # called Relu because of the checkpoint :)
            self.DenseReluDense = DenseRelu(embed_size = embed_dim, ff_size = ff_dim)

        self.layer_norm = RMSNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, key_pad_mask = None):

        x = self.layer_norm(x)
        
        if self.attention:
            x = self.SelfAttention(x, key_pad_mask = key_pad_mask)

        else: x = self.DenseReluDense(x)

        x = self.dropout(x)

        return x

class T5Encoder(nn.Module):
    def __init__(self, embed_dim, ff_dim, index):
        super().__init__()
        self.layer = nn.ModuleList([T5EncoderLayer(embed_dim, ff_dim, index = index, attention = True), T5EncoderLayer(embed_dim, ff_dim, index = 1)])

    def forward(self, x, key_pad_mask = None):

        attn_outputs = self.layer[0](x, key_pad_mask = key_pad_mask)
        x = self.layer[1](x)

        x += attn_outputs
        
        return x

class T5EncoderModel(nn.Module):
    def __init__(self, embed_dim: int = 768, ff_dim: int = 2048):
        super().__init__()
        vocab_size = 32128

        self.encoder = nn.Module()
        self.encoder.block = nn.ModuleList([T5Encoder(embed_dim, ff_dim, index = i) for i in range(12)])
        self.encoder.final_layer_norm = RMSNorm(embed_dim)

        self.encoder.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.shared = nn.Linear(embed_dim, vocab_size, bias = False)

        self.shared.weight = self.encoder.embed_tokens.weight # just in case

    def forward(self, x, attn_mask: torch.Tensor, return_hidden: bool = False):

        src_key_mask = attn_mask == 0

        x = self.encoder.embed_tokens(x) 
        
        for block in self.encoder.block:
            x = block(x, key_pad_mask = src_key_mask)

        x = self.encoder.final_layer_norm(x)

        # for testing
        if return_hidden:
            return x

        x = self.shared(x)

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
    tokens, attn = tokenizer.encode("hello world")
    tokens2, attn2 = tokenizer.encode("hello worlds")

    model = T5EncoderModel()
    model = load_t5(model)

    import torch.nn.functional as F

    outputs = F.normalize(model(tokens.unsqueeze(0), attn.unsqueeze(0), return_hidden = True).mean(dim = 1))
    outputs2 = F.normalize(model(tokens2.unsqueeze(0), attn2.unsqueeze(0), return_hidden = True).mean(dim = 1))
    
    sim = F.cosine_similarity(outputs, outputs2)

    print(sim)

if __name__ == "__main__":
    test_t5()
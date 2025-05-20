############################################################################## REIMPLEMENTATION OF OPENAI CLIP ############################################################################

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import TorchTokenizer

def create_causal_mask(input_shape, device: str = "cpu"):
    
    B, seq_len, _ = input_shape.size()
    mask = torch.full((seq_len, seq_len),
                      torch.finfo(input_shape.dtype).min, # smallest representable number (-max)
                      device = device)

    # zero out the lower triangle
    # the same as torch.triu(mask, diagonal = 1)
    mask_cond = torch.arange(seq_len, device = device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(seq_len, 1), 0)

    # for every batch to share the mask
    # add two new dims          expand to
    return mask[None, None, :, :].expand(B, 1, seq_len, seq_len)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class TextEncoder(nn.Module):
    def __init__(self, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, max_seq_len: int = 77):
        super(TextEncoder, self).__init__()

        vocab_size = 49408

        self.embeddings = nn.Module()
        self.embeddings.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # tokenizer's context_length must be set to 77 tokens for CLIP to work
        self.embeddings.position_embedding = nn.Embedding(max_seq_len, embed_dim) # 77 = context length

        self.encoder = Encoder(embed_size = embed_dim, depth = depth, num_heads = num_heads)

        self.final_layer_norm = nn.LayerNorm(embed_dim)

        # add in the register buffer so we won't have to recompute it every forward pass
        self.register_buffer(
            "positions", torch.arange(max_seq_len)[None, :], persistent = False
        )

    def forward(self, text: torch.Tensor, attention_mask: torch.Tensor, return_hidden: bool = False):

        x = self.embeddings.token_embedding(text.long())
        pos_ids = self.positions[:, :x.size(1)] # till seq_length

        pos_embed = self.embeddings.position_embedding(pos_ids)

        x = x + pos_embed.to(x.dtype).to(x.device)
        causal_mask = create_causal_mask(x, device = x.device)

        # obtain text embeddings
        x, pentlum = self.encoder(x, 
                                  attention_mask,
                                  causal_mask = causal_mask,
                                  return_pentlum = True)
        
        # for debugging
        if return_hidden:
            return x
        
        x = self.final_layer_norm(x)

        # ensure batch dim
        if x.dim() == 2: x = x.unsqueeze(0)

        # for each batch, get the last token (eos)
        pooled = x[torch.arange(x.size(0)), text.argmax(dim = -1)]

        return pooled, pentlum
    
class CLIPAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be dividable by num_heads"

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5 # reciporical of sqrt
        
        self.out_proj = nn.Linear(embed_dim, output_dim if output_dim is not None else embed_dim)

    def forward(self, x, attention_mask, causal_attention_mask):

        B, seq_len, embed_dim = x.size()

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = query.view(B, seq_len, -1, self.head_dim).transpose(1, 2)
        key = key.view(B, seq_len, -1, self.head_dim).transpose(1, 2)
        value = value.view(B, seq_len, -1, self.head_dim).transpose(1, 2)

        # combine the masks
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        attention_mask = attention_mask + causal_attention_mask

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attn_weights = attn_weights + attention_mask

        attn_outputs = F.softmax(attn_weights, dim = -1, dtype = torch.float32).to(query.dtype)

        attn_output = torch.matmul(attn_outputs, value).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, seq_len, embed_dim).contiguous()

        output = self.out_proj(attn_output)

        return output
    
class MLP(nn.Module):
    def __init__(self, embed_size, ratio = 4):
        super().__init__()

        self.fc1 = nn.Linear(embed_size, embed_size * ratio)
        self.fc2 = nn.Linear(embed_size * ratio, embed_size)
        self.gelu = QuickGELU()

    def forward(self, x: torch.Tensor):

        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)

        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, embed_size: int = 768, ratio: int = 4, num_heads: int = 12):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size = embed_size, ratio = ratio)

        self.self_attn = CLIPAttention(num_heads = num_heads, embed_dim = embed_size)

    def forward(self, x: torch.Tensor, src_pad_key = None, causal_mask = None):
        
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attn(x, attention_mask = src_pad_key, causal_attention_mask = causal_mask)

        # normalize and apply residual connections
        x = x + residual

        residual = x
        x = self.layer_norm2(x)
        
        x = self.mlp(x)
        x = x + residual

        return x

class Encoder(nn.Module):
    def __init__(self, embed_size = 768, depth: int = 12, num_heads: int = 12):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(embed_size = embed_size, num_heads = num_heads) for _ in range(depth)])

    def forward(self, hidden_states: torch.Tensor, attention_mask = None, causal_mask = None, return_pentlum: bool = False):

        if attention_mask is not None:

            src_key_mask = attention_mask == 0
            total_layers = len(self.layers)

            for i, layer in enumerate(self.layers):
                hidden_states = layer(hidden_states, src_key_mask, causal_mask = causal_mask)
                # get the before the last hidden state for DiT
                if i == (total_layers - 2) and return_pentlum:
                    pentlum = hidden_states

        if return_pentlum: return hidden_states, pentlum
        else: return hidden_states

class CLIP(nn.Module):
    def __init__(self, embed_dim: int = 768, depth: int = 12, num_heads: int = 12):
        super(CLIP, self).__init__()

        self.tokenizer = TorchTokenizer()
        self.text_model = TextEncoder(embed_dim = embed_dim, depth = depth, num_heads = num_heads,
                                      max_seq_len = self.tokenizer.max_length)
                
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias = False)
        self.text_model.eval()

    def forward(self):
        pass
    
    def encode_text(self, input_ids, attention_mask = None):
        """ Tokenize (if needed) and encode texts, returning embeddings and mask. Function for ConditionalPromptNorm """
        
        # tokenize strings if raw text passed
        if attention_mask is None:
            input_ids, attention_mask = self.tokenizer.tokenize_batch([input_ids])
        print(input_ids)
        print(attention_mask)
        # ensure batch dim
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        with torch.no_grad():
            pooled_emb, pentlum = self.text_model(input_ids.long(), attention_mask)

        pooled_emb = self.text_projection(pooled_emb)

        pooled_emb = F.normalize(pooled_emb, dim=-1)

        return pooled_emb, pentlum

class OpenCLIP(CLIP):
    def __init__(self):
        super().__init__(embed_dim = 1280, depth = 32, num_heads = 16)

def load_clip(model: CLIP, model_2: OpenCLIP, device = "cpu"):

    DEBUG = False

    # checkpoints could be installed automatically from encoders/get_checkpoints.py

    clip_path = os.path.join(os.getcwd(), "encoders", "hub", "checkpoints", "clip_model.pth")
    model.load_state_dict(torch.load(clip_path, map_location = device), strict = True)

    path = "d:\\encoders\\clip_2_saved\\saved.pth"
    path2 = "d:\\encoders\\clip_2\\models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k\\snapshots\\743c27bd53dfe508a0ade0f50698f99b39d03bec\\pytorch_model-00002-of-00002.bin"

    #missing, unexpected = model_2.load_state_dict(torch.load(path, map_location = device), strict = False)
    #missing2, unexpected2 = model_2.load_state_dict(torch.load(path2, map_location = device), strict = False)
    
    # for debuggging
    if DEBUG:
        print(f"Missing keys ({len(missing)}):", missing)
        print(f"\nUnexpected keys ({len(unexpected)}):", unexpected)

    model.eval()
    model_2.eval()

    return model, model_2

def test_clip():
 
    model = CLIP()
    model_2 = OpenCLIP()

    model, model_2 = load_clip(model, model_2)

    ids, _ = model.encode_text("a photo of a cat")
    ids2, _ = model.encode_text("a photo of a dog")

    from torch.nn.functional import cosine_similarity
    print("cosine cat/dog:", cosine_similarity(ids, ids2, dim=-1).item())

if __name__ == "__main__":
    test_clip()
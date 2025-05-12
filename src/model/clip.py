############################################################################## REIMPLEMENTATION OF OPENAI CLIP ############################################################################

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import TorchTokenizer

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
class TextEncoder(nn.Module):
    def __init__(self, embed_dim: int = 768, depth: int = 12, num_heads: int = 12):
        super(TextEncoder, self).__init__()

        vocab_size = 49408

        self.embeddings = nn.Module()
        self.embeddings.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # tokenizer's context_length must be set to 77 tokens
        self.embeddings.position_embedding = nn.Embedding(77, embed_dim) # 77 = context length

        self.encoder = Encoder(embed_size = embed_dim, depth = depth, num_heads = num_heads)

        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, text: torch.Tensor, attention_mask: torch.Tensor, return_hidden: bool = False):

        x = self.embeddings.token_embedding(text.long())

        #                       seq_length
        positions = torch.arange(x.size(1))
        pos_embed = self.embeddings.position_embedding(positions)

        x += pos_embed.to(x.dtype).to(x.device)

        # obtain text embeddings
        x = x.permute(1, 0, 2)
        x, pentlum = self.encoder(x, attention_mask, return_pentlum = True)
        x = x.permute(1, 0, 2)
        
        # for debugging
        if return_hidden:
            return x
        
        x = self.final_layer_norm(x)

        # ensure batch dim
        if x.dim() == 2: x = x.unsqueeze(0)

        # for each batch, get the last token (eos)
        pooled = x[torch.arange(x.size(0)), text.argmax(dim = -1)]

        return pooled, pentlum
    
class AttentionPool2d(nn.Module):
    # modified class from: https://github.com/openai/CLIP/blob/main/clip/model.py#L58
    def __init__(self, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.num_heads = num_heads
        
        if output_dim is None: self.out_proj = nn.Linear(embed_dim, embed_dim)
        else: self.out_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, x, src_pad_key = None, use_self_attention: bool = True):

        # ensure (seq_len, B, embed_dim)
        if x.shape[0] == 1:
            x = x.permute(1, 0, 2)

        if use_self_attention: 
            query = x
        else: query = x[:1] # pooled attention

        x, _ = F.multi_head_attention_forward(
            query = query, key = x, value = x,
            embed_dim_to_check = x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            use_separate_proj_weight=True,
            training = False,
            need_weights = False,
            key_padding_mask = src_pad_key
        )

        # (B, C)
        return x.squeeze(0)
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

        self.self_attn = AttentionPool2d(num_heads = num_heads, embed_dim = embed_size)

    def forward(self, x: torch.Tensor, src_pad_key = None):
        
        residual = x
        x = self.layer_norm1(x)
        
        if src_pad_key is not None: 
            x = self.self_attn(x, src_pad_key = src_pad_key, use_self_attention = True)

        # normalize and apply residual connections
        x += residual

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x += residual

        return x

class Encoder(nn.Module):
    def __init__(self, embed_size = 768, depth: int = 12, num_heads: int = 12):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(embed_size = embed_size, num_heads = num_heads) for _ in range(depth)])

    def forward(self, x: torch.Tensor, attention_mask = None, return_pentlum: bool = False):

        if attention_mask is not None:
            src_key_mask = attention_mask == 0
            if src_key_mask.dim() == 1: src_key_mask = src_key_mask.unsqueeze(0)

            total_layers = len(self.layers)

            for i, layer in enumerate(self.layers):
                x = layer(x, src_key_mask)
                if i == (total_layers - 2) and return_pentlum:
                    pentlum = x
        
        else:
            for layer in self.layers:
                x = layer(x)

        if return_pentlum: return x, pentlum
        else: return x
class CLIP(nn.Module):
    def __init__(self, embed_dim: int = 768, depth: int = 12, num_heads: int = 12):
        super(CLIP, self).__init__()

        self.text_model = TextEncoder(embed_dim = embed_dim, depth = depth, num_heads = num_heads)
        self.tokenizer = TorchTokenizer()
        
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias = False)
        self.text_model.eval()

    def forward(self):
        pass
    
    def encode_text(self, input_ids, attention_mask = None):
        """ Tokenize (if needed) and encode texts, returning embeddings and mask. Function for ConditionalPromptNorm """
        
        # tokenize strings if raw text passed
        if attention_mask is None:
            input_ids, attention_mask = self.tokenizer.tokenize(input_ids)
        
        # ensure batch dim
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        with torch.no_grad():
            pooled_emb, pentlum = self.text_model(input_ids.long(), attention_mask)

        pooled_emb = self.text_projection(pooled_emb)

        pooled_features = F.normalize(pooled_emb, dim=-1)

        return pooled_features, pentlum

class OpenCLIP(CLIP):
    def __init__(self):
        super().__init__(embed_dim = 1280, depth = 32, num_heads = 16)

def load_clip(model: CLIP, model_2: OpenCLIP, device = "cpu"):

    DEBUG = False

    # checkpoints could be installed automatically from encoders/get_checkpoints.py

    clip_path = os.path.join(os.getcwd(), "encoders", "hub", "checkpoints", "clip_model.pth")
    #model.load_state_dict(torch.load(clip_path, map_location = device), strict = True)

    path = "d:\\encoders\\clip_2_saved\\saved.pth"
    path2 = "d:\\encoders\\clip_2\\models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k\\snapshots\\743c27bd53dfe508a0ade0f50698f99b39d03bec\\pytorch_model-00002-of-00002.bin"

    missing, unexpected = model_2.load_state_dict(torch.load(path, map_location = device), strict = False)
    missing2, unexpected2 = model_2.load_state_dict(torch.load(path2, map_location = device), strict = False)
    
    # for debuggging
    if DEBUG:
        print(f"Missing keys ({len(missing2)}):", missing2)
        print(f"\nUnexpected keys ({len(unexpected2)}):", unexpected2)

    model.eval()
    model_2.eval()

    return model, model_2

def test_clip():
 
    model = CLIP()
    model_2 = OpenCLIP()

    model, model_2 = load_clip(model, model_2)

    ids, _ = model_2.encode_text("a photo of a cat")
    ids2, _ = model_2.encode_text("a photo of a dog")
    
    from torch.nn.functional import cosine_similarity

    print("cosine cat/dog:", cosine_similarity(ids, ids2, dim=-1).item())

if __name__ == "__main__":
    test_clip()
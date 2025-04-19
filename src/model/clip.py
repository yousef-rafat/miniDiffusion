############################################################################## REIMPLEMENTATION OF OPENAI CLIP ############################################################################

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import TorchTokenizer
from torchvision.transforms.v2 import Resize

class TextEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super(TextEncoder, self).__init__()

        vocab_size = 49408

        self.embeddings = nn.Module()
        self.embeddings.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # tokenizer's context_length must be set to 77 tokens
        self.embeddings.position_embedding = nn.Embedding(77, embed_dim) # 77 = context length

        self.encoder = Encoder(embed_size = embed_dim)

        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):

        x = self.embeddings.token_embedding(x.long())

        #                       seq_length
        positions = torch.arange(x.size(1))
        pos_embed = self.embeddings.position_embedding(positions)

        x += pos_embed

        # obtain text embeddings
        x = x.permute(1, 0, 2)
        x = self.encoder(x, attention_mask)
        x = x.permute(1, 0, 2)

        # ensure batch dim
        if x.dim() == 2: x = x.unsqueeze(0)
        if attention_mask.dim() == 1: attention_mask = attention_mask.unsqueeze(0)

        # get the length of the valid tokens (non padded)
        token_lens = attention_mask.sum(dim = 1)
        # get the last token so we can get EOS token
        inds = token_lens - 1
        # for each batch, get the last token
        x = x[torch.arange(x.size(0)), inds]

        return self.final_layer_norm(x)
    
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

    def forward(self, x, src_pad_key = None, text = False):

        # ensure (seq_len, B, embed_dim)
        if x.shape[0] == 1:
            x = x.permute(1, 0, 2)

        if text: query = x
        else: query = x[:1]

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
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor):

        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)

        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, embed_size: int = 768, ratio: int = 4, num_heads: int = 8):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size = embed_size, ratio = ratio)

        self.self_attn = AttentionPool2d(num_heads = num_heads, embed_dim = embed_size)

    def forward(self, x: torch.Tensor, src_pad_key = None):
        
        if src_pad_key is not None: attn_out = self.self_attn(x, src_pad_key = src_pad_key, text = True)
        else: attn_out = self.self_attn(x)

        # normalize and apply residual connections
        x = self.layer_norm1(x)
        x += attn_out
        x = self.layer_norm2(x)
        x += self.mlp(x)

        return x

class Encoder(nn.Module):
    def __init__(self, embed_size = 768):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(embed_size = embed_size) for _ in range(12)])

    def forward(self, x: torch.Tensor, attention_mask = None):

        if attention_mask is not None:
            src_key_mask = attention_mask == 0
            if src_key_mask.dim() == 1: src_key_mask = src_key_mask.unsqueeze(0)

            for layer in self.layers:
                x = layer(x, src_key_mask)
        
        else:
            for layer in self.layers:
                x = layer(x)

        return x

class ImageEncoder(nn.Module):
    def __init__(self, project_dim: int = 768, embed_dim: int = 768):
        super(ImageEncoder, self).__init__()

        self.embeddings = nn.Module()
        self.embeddings.class_embedding = nn.Parameter(torch.randn(embed_dim))

        # slide over each kernel with stride and kernel size of 32 (typically no bias is used in these convs)
        self.embeddings.patch_embedding = nn.Conv2d(in_channels = 3, out_channels = project_dim, kernel_size = 32, stride = 32, bias = False)
        self.embeddings.position_embedding = nn.Embedding(50, embed_dim)

        self.encoder = Encoder()
        
        self.pre_layrnorm = nn.LayerNorm(project_dim)
        self.post_layernorm = nn.LayerNorm(project_dim)

    def forward(self, x: torch.Tensor):
        
        # output: (B, embed_dim, H_, W_)
        if x.dim() == 5: x = x.squeeze(0)
        x = self.embeddings.patch_embedding(x)
        x = x.flatten(2) # (B, embed_dim, HW)

        # (B, HW, embed_dim)
        x = x.transpose(1, 2)

        cls_tokens = self.embeddings.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        positions = torch.arange(x.size(1), device = x.device)
        pos_embed = self.embeddings.position_embedding(positions)

        x += pos_embed

        x = self.pre_layrnorm(x)

        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x.permute(1, 0, 2) 

        # get cls token
        x = x[:, 0, :]

        x = self.post_layernorm(x)

        return x
    
class CLIP(nn.Module):
    def __init__(self, project_dim: int = 768, embed_dim: int = 512):
        super(CLIP, self).__init__()

        self.vision_model = ImageEncoder(project_dim = project_dim)
        self.text_model = TextEncoder(embed_dim = embed_dim)
        self.tokenizer = TorchTokenizer()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.7) 
        self.visual_projection = nn.Linear(project_dim, embed_dim, bias = False)
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias = False)

        self.vision_model.eval()
        self.text_model.eval()

    def forward(self, image: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:

        " Compute the relationship between image and text  "

        # get fixed size to comply with the checkpoint position_embeddings nn.Embedding(50, embed_dim)
        image = Resize(size=(224, 224))(image)

        image_features = self.vision_model(image)

        # projections
        text_features = self.text_projection(text_embed)
        image_features = self.visual_projection(image_features)
        
        # normalization
        text_features = F.normalize(text_features, dim = -1)
        image_features = F.normalize(image_features, dim = -1)

        logits = self.logit_scale.exp() * (image_features @ text_features.t())

        return logits
    
    def encode_text(self, input_ids, attention_mask = None):
        """ Tokenize (if needed) and encode texts, returning embeddings and mask. Function for ConditionalPromptNorm """

        # tokenize strings if raw text passed
        if attention_mask is None:
            input_ids, attention_mask = self.tokenizer.tokenize(input_ids)
        
        # ensure batch dim
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        with torch.no_grad():
            text_emb = self.text_model(input_ids.long(), attention_mask)

        return text_emb

def load_clip(model: CLIP, device = "cpu"):

    # checkpoints could be installed automatically from encoders/get_checkpoints.py

    clip_path = os.path.join(os.getcwd(), "encoders", "hub", "checkpoints", "clip_model.pth")
    missing, unexpected = model.load_state_dict(torch.load(clip_path, map_location = device), strict = True)

    # for debuggging
    if len(missing) != 0:
        print(f"Missing keys ({len(missing)}):", missing)
        print(f"\nUnexpected keys ({len(unexpected)}):", unexpected)

    model.eval()

    return model

def test_clip():

    from PIL import Image
    from torchvision.transforms import ToTensor

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])
 
    model = CLIP()

    image_path = os.path.join(os.getcwd(), "assets", "cat.webp")

    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)
    #image = ToTensor()(image).unsqueeze(0)

    model = load_clip(model)

    ids = model.encode_text("a photo of a cat")

    # run forward
    logit = model(image, ids) 
    print("Cat similarity:", logit)

    # same for dog
    ids2 = model.encode_text("a photo of a dog")
    logit2 = model(image, ids2)
    print("Dog similarity:", logit2)

if __name__ == "__main__":
    test_clip()
import os
import torch
import torchvision
import torch.nn as nn
from dit_components import patchify
from torchvision.models import ViT_B_16_Weights
from attention import PagedTransformerEncoderLayer

class TextEncoder(nn.Module):
    def __init__(self, project_dim, embed_dim):
        super(TextEncoder, self).__init__()

        from tokenizer import TorchTokenizer # avoid circular import error

        self.model = PagedTransformerEncoderLayer(embed_dim = 512, num_heads = 8)

        self.tokenizer = TorchTokenizer()

        self.projection = nn.Linear(embed_dim, project_dim)
        self.layer_norm = nn.LayerNorm(project_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):

        # obtain text embeddings
        x = self.model(input_ids, src_key_padding_mask = (attention_mask == 0) )

        # get classification token
        x = x[:, 0, :]

        x = self.projection(x)

        return self.layer_norm(x)
    
    def encoder_tokenize(self, prompt: str):
        return self.tokenizer(prompt, padding = True, truncation = True, max_length = 1024)

class ImageEncoder(nn.Module):
    def __init__(self, project_dim, size: int = 16):
        super(ImageEncoder, self).__init__()

        model_path = os.path.join(os.getcwd(), "encoders", "hub", "checkpoints", "vit_b_16_lc_swag-4e70ced5.pth")

        if not os.path.exists(model_path): self.model = torchvision.models.vit_b_16(weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        else: 
            # initate model and load state
            self.model = torchvision.models.vit_b_16()
            self.model.load_state_dict(torch.load(model_path))

        self.projection = nn.Linear(self.model.hidden_dim, project_dim)
        self.layer_norm = nn.LayerNorm(project_dim)
        self.size = size

    def forward(self, x: torch.Tensor):

        x = patchify(x, size = self.size, stride = self.size)

        x = self.model(x)
        x = self.projection(x)

        return self.layer_norm(x)
    
class CLIP(nn.Module):
    def __init__(self, project_dim: int = 512, embed_dim: int = 512):
        super(CLIP, self).__init__()

        os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), "encoders")

        self.image_encoder = ImageEncoder(project_dim = project_dim)
        self.text_encoder = TextEncoder(project_dim = project_dim, embed_dim = embed_dim)
        
        self.temp = nn.Parameter(torch.ones([]) * 0.7)

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):

        " Compute the relationship between image and text  "

        image_features = self.image_encoder(image)
        text_features = self.text_encoder(input_ids, attention_mask)

        image_features = image_features / image_features.norm(dim = -1, keepdim = True)
        text_features = text_features / text_features.norm(dim = -1, keepdim = True)        

        logits = self.temp.exp() * (image_features @ text_features.t())

        return logits

def test_clip(mask = True):

    model = CLIP()
    image = torch.rand(3, 224, 224)
    input_ids = torch.randint(50000, size = (1024, 512))

    if mask: attn_mask = torch.zeros(1024, 512)

    outputs = model(image = image.unsqueeze(0), input_ids = input_ids, attention_mask = attn_mask)
    print(outputs)

test_clip()
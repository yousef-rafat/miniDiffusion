import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from model.tokenizer import TorchTokenizer
from torchvision.models import ViT_B_16_Weights
from model.attention import PagedTransformerEncoderLayer

class TextEncoder(nn.Module):
    def __init__(self, project_dim, embed_dim: int = 512, heads: int = 8, depth: int = 5):
        super(TextEncoder, self).__init__()

        self.model = nn.ModuleList(
            [PagedTransformerEncoderLayer(num_heads = heads, embed_dim =  embed_dim) for _ in range(depth)]
        )

        self.tokenizer = TorchTokenizer()

        self.projection = nn.Linear(embed_dim, project_dim)
        self.layer_norm = nn.LayerNorm(project_dim)

        vocab_size = 50262
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):

        x = self.token_embedding(x.long())

        # obtain text embeddings
        for layer in self.model:
            x = layer(x, src_key_padding_mask = (attention_mask == 0).float())

        if x.dim() == 2: x = x.unsqueeze(0)
        # get classification token
        x = x[:, 0, :]

        x = self.projection(x)

        return self.layer_norm(x)

class ImageEncoder(nn.Module):
    def __init__(self, project_dim, size: int = 222):
        super(ImageEncoder, self).__init__()

        model_path = os.path.join(os.getcwd(), "encoders", "hub", "checkpoints", "vit_b_16_lc_swag-4e70ced5.pth")

        if os.path.exists(model_path): 
            self.model = torchvision.models.vit_b_16(weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        else: 
            # initate model and load state
            self.model = torchvision.models.vit_b_16()
            self.model.load_state_dict(torch.load(model_path))

        # change input layer to accept latents
        self.model.image_size = size

        # correct positional embedding
        num_patches = (size // 16) ** 2 
        hidden_dim = self.model.hidden_dim
        self.model.encoder.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))

        self.projection = nn.Linear(1000, project_dim)
        self.layer_norm = nn.LayerNorm(project_dim)
        self.size = size

    def forward(self, x: torch.Tensor):

        x = self.model(x)
        x = self.projection(x)

        return self.layer_norm(x)
    
class CLIP(nn.Module):
    def __init__(self, project_dim: int = 512, embed_dim: int = 512, size: int = 222):
        super(CLIP, self).__init__()

        os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), "encoders")

        self.image_encoder = ImageEncoder(project_dim = project_dim, size = size)
        self.text_encoder = TextEncoder(project_dim = project_dim, embed_dim = embed_dim)
        
        self.temp = nn.Parameter(torch.ones([]) * 0.7)
        self.project_dim = project_dim

        self.image_encoder.eval()
        self.text_encoder.eval()

    def adjust_image_size(self, size: int):
        self.image_encoder = ImageEncoder(self.project_dim, size = size)

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

        " Compute the relationship between image and text  "

        image_features = F.normalize(self.image_encoder(image), dim = -1)

        #if input_ids.dim() != 3: text_features = F.normalize(self.text_encoder(input_ids, attention_mask), dim = -1)
        #else: 
            # masked mean pooling so only non-padded tokens contribute

        if input_ids.dim() == 4: input_ids = input_ids.squeeze(1)

        text_features = input_ids.float() * attention_mask.unsqueeze(-1)
        text_features = text_features.sum(dim = 1) / attention_mask.sum(dim = 1, keepdim = True)

        image_features = image_features / image_features.norm(dim = -1, keepdim = True)
        text_features = text_features / text_features.norm(dim = -1, keepdim = True)  

        logits = self.temp.exp() * (image_features @ text_features.t())

        return logits
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """ Encodes tokens for ConditionalPromptNorm """

        with torch.no_grad():
            text_embeddings = self.text_encoder(input_ids, attention_mask)

        return text_embeddings


def test_clip(mask = True):

    model = CLIP()
    image = torch.rand(3, 224, 224)
    input_ids = torch.randint(50000, size = (1024, 512))

    if mask: attn_mask = torch.zeros(1024, 512)

    outputs = model(image = image.unsqueeze(0), input_ids = input_ids.unsqueeze(0), attention_mask = attn_mask.unsqueeze(0))
    print("output: ", outputs)
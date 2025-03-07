import torch
import torchvision
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights

class TextEncoder(nn.Module):
    def __init__(self, project_dim, embed_dim):
        super(TextEncoder, self).__init__()

        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased', output_attentions=True)
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')

        self.projection = nn.Linear(embed_dim, project_dim)
        self.layer_norm = nn.LayerNorm(project_dim)

    def forward(self, input_ids, attention_mask):
        x = self.model(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state

        # get classification token
        x = x[:, 0, :]

        x = self.projection(x)

        return self.layer_norm(x)
    
    def encoder_tokenize(self, prompt: str):
        return self.tokenizer(prompt, padding = True, truncation = True, max_length = 1024)

class ImageEncoder(nn.Module):
    def __init__(self, project_dim):
        super(ImageEncoder, self).__init__()

        self.model = torchvision.models.vit_b_16(weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)

        self.projection = nn.Linear(self.model.hidden_dim, project_dim)
        self.layer_norm = nn.LayerNorm(project_dim)

    def forward(self, x: torch.Tensor):

        x = self.model(x)
        x = self.projection(x)

        return self.layer_norm(x)
    
class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        
        self.temp = nn.Parameter(torch.ones([]) * 0.7)

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):

        " Compute the relationship between image and text  "

        image_features = self.image_encoder(image)
        text_features = self.text_encoder(input_ids, attention_mask)

        image_features = image_features / image_features.norm(dim = -1, keepdim = True)
        text_features = text_features / text_features.norm(dim = -1, keepdim = True)        

        logits = self.temp.exp() * (image_features @ text_features.T)

        return logits


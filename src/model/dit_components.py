import os
import json
import torch
import torch.nn as nn
from clip import CLIP
from torch import Tensor

def patchify(x: Tensor, size: int, stride: int):
    " Turn a latent into patches "
    # usually, the smaller the patch size the better quality of the generated image
    # if you have enough gpu power, try training with patch size of 2
    batch, channels, height, width = x.shape

    assert height % 4 == 0 and width % 4 == 0, "Image must be divisible into 16 patches"
    size = height // 4  # Ensuring 4x4 patches

    stride = size  # no overlap

    patches = x.unfold(2, size, stride).unfold(3, size, stride)

    # (batch, num_patches, channels, patch_size, patch_size)
    patches = patches.contiguous().view(batch, -1, channels, size, size)

    return patches

def depatchify(x: Tensor, img_size: int) -> torch.Tensor:
    " turn patches into an image "

    # check if image is 4 by 4 patches
    batch, patches, channels, size, _ = x.shape

    grid_size = int(patches * 0.5)
    assert grid_size * size == img_size, "Image must be 4x4 patches"

    # turning patches into images 
    x = x.view(batch, grid_size, grid_size, channels, size, size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(batch, channels, img_size, img_size) 

    return x

class TorchTokenizer:
    # TODO: add special tokens to tokenizer and remove detokenizer
    def __init__(self, tokenizer_path="tokenizer.json"):
        # Load the tokenizer.json file
        file_path = os.path.join(os.getcwd(), "src", "model", tokenizer_path)

        with open(file_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)
        
        # Extract vocabulary
        self.vocab = tokenizer_data["model"]["vocab"]
        # For detokenization
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        self.vocab_len = len(self.vocab)
        self.special_vocab = { '<pad>': self.vocab_len + 1, '<start>': self.vocab_len + 2, '<end>': self.vocab_len + 3, '<unk>': self.vocab_len + 4 }

        self.start = self.special_vocab['<start>']
        self.end = self.special_vocab['<end>']
        self.pad_id = self.special_vocab['<pad>']

        self.unk_token = self.special_vocab['<unk>']

    def tokenize(self, text):
        """Tokenizes text by splitting on whitespace and maps to vocab."""
        # split by white space
        tokens = text.lower().split()
        token_ids = [self.vocab.get(token, self.unk_token) for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long)

    def detokenize(self, token_ids, skip_special_tokens = True):
        """Converts token IDs back to text."""
        tokens = [self.inv_vocab.get(id, self.unk_token) for id in token_ids]
        return " ".join(tokens)

    def pad_sequence(self, token_ids, max_length):
        """Pads tokenized sequences to a fixed length."""
        pad_length = max_length - len(token_ids)
        if pad_length > 0:
            token_ids = torch.cat([token_ids, torch.full((pad_length,), self.pad_id, dtype=torch.long)])
        return token_ids[:max_length]

# Example usage
tokenizer = TorchTokenizer()
text = "hello world"
token_ids = tokenizer.tokenize(text)

print("Tokenized:", token_ids) 
print("Detokenized:", tokenizer.detokenize(token_ids.tolist()))
        

class ConditionalPromptNorm(nn.Module):
    # normalization with Feed-Forward layer for text prompts
    # should be encoded with clip
    def __init__(self, hidden_size, dim: int):
        super(ConditionalPromptNorm, self).__init__()

        # normalization
        # elementwise_affine will put biases into zeros and weights into ones
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine = False)
        # FF layer
        self.fcw = nn.Linear(dim, hidden_size)
        self.fcb = nn.Linear(dim, hidden_size)

    def forward(self, x: Tensor, features):   
        bs = x.size(0) # batch size

        out = self.norm(x)
        w = self.fcw(features).reshape(bs, 1, -1)
        b = self.fcb(features).reshape(bs, 1, -1)

        return  w * out + b

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, device = 'cpu', dropout = 0.1):
        super(RotaryPositionalEncoding, self).__init__()

        assert d_model % 2 == 0, "d_model must be even for RoPE"

        self.dropout = nn.Dropout(p = dropout)

        # Create a rotation matrix.
        self.rotation_matrix = torch.zeros(d_model, d_model, device = device)
        for i in range(d_model):
            for j in range(d_model):
                self.rotation_matrix[i, j] = torch.cos(i * j * 0.01)

        # Create a positional embedding matrix.
        self.positional_embedding = torch.zeros(max_seq_len, d_model, device = device)
        for i in range(max_seq_len):
            for j in range(d_model):
                self.positional_embedding[i, j] = torch.cos(i * j * 0.01)

    def forward(self, x):
        """ applies rotational encoding """

        # Add the positional embedding to the input tensor.
        x += self.positional_embedding

        # Apply the rotation matrix to the input tensor.
        x = torch.matmul(x, self.rotation_matrix)

        return self.dropout(x)
    
class HandlePrompt(nn.Module):
    # be used in training with raw string data
    def __init__(self):
        super(HandlePrompt, self).__init__()

        self.tokenizer = TorchTokenizer()
        self.norm = ConditionalPromptNorm()

    def forward(self, prompt: str):
        x = self.tokenizer.tokenize(prompt)
        x = self.norm(x)

        return x
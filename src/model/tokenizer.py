import os
import json
import torch
from itertools import takewhile

class TorchTokenizer:
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

        self.inv_vocab.update({v: k for k, v in self.special_vocab.items()})

        self.start = self.special_vocab['<start>']
        self.end = self.special_vocab['<end>']
        self.pad_id = self.special_vocab['<pad>']

        self.unk_token = self.special_vocab['<unk>']

    def tokenize(self, text, max_length: int = 512):
        """Tokenizes text by splitting on whitespace and maps to vocab."""
        # split by white space
        tokens = text.lower().split()
        token_ids = [self.vocab.get(token, self.unk_token) for token in tokens]
        attention_mask = torch.ones(max_length)

        # truncate
        if len(token_ids) < max_length - 2:
            token_ids = token_ids[:max_length - 2]

        # add starting token
        token_ids.insert(0, self.start)

        # add ending token 
        token_ids.append(self.end)

        # pad the rest
        while len(token_ids) < max_length:
            token_ids.append(self.pad_id)
            attention_mask[len(token_ids) - 1] = 0

        return torch.tensor(token_ids, dtype = torch.long), attention_mask.long()

    def detokenize(self, token_ids, skip_special_tokens = True):
        """Converts token IDs back to text."""
        
        #                                                          skip pad tokens
        tokens = [self.inv_vocab.get(id, self.unk_token) for id in takewhile(lambda x: x != self.pad_id, token_ids)]

        if skip_special_tokens:
            special_tokens = set(self.special_vocab.keys())
            tokens = [token for token in tokens if token not in special_tokens]

        return " ".join(tokens)
    
    
def test_tokenizer():
    tokenizer = TorchTokenizer()
    text = "hello world"
    token_ids = tokenizer.tokenize(text)

    print("Tokenized:", token_ids) 
    print("Detokenized:", tokenizer.detokenize(token_ids.tolist()))
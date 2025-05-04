import os
import json
import torch
from itertools import takewhile

class TorchTokenizer:
    def __init__(self, tokenizer_path: str = "tokenizer.json", max_length: int = 77):

        # Load the tokenizer.json file
        file_path = os.path.join(os.getcwd(), "src", "model", tokenizer_path)

        with open(file_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)
        
        self.max_length = max_length
        # Extract vocabulary
        self.vocab = tokenizer_data["model"]["vocab"]
        # For detokenization
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        self.vocab_len = len(self.vocab)

        self.start = tokenizer_data['added_tokens'][0]['id']
        self.end = tokenizer_data['added_tokens'][1]['id']

        self.pad_id = self.unk_token = self.end

        self.special_vocab = ["<|startoftext|>", "<|endoftext|>"]

    def tokenize(self, text):
        """Tokenizes text by splitting on whitespace and maps to vocab."""
        # split by white space
        tokens = text.lower().split()
        token_ids = [self.vocab.get(token, self.unk_token) for token in tokens]
        attention_mask = torch.ones(self.max_length)

        # truncate
        if len(token_ids) < self.max_length:
            token_ids = token_ids[:self.max_length]

        # add starting token
        token_ids.insert(0, self.start)

        # add ending token 
        token_ids.append(self.end)

        # pad the rest
        while len(token_ids) < self.max_length:
            token_ids.append(self.pad_id)
            attention_mask[len(token_ids) - 1] = 0

        return torch.tensor(token_ids, dtype = torch.long), attention_mask.long()

    def detokenize(self, token_ids, skip_special_tokens = True):
        """Converts token IDs back to text."""
        
        #                                                          skip pad tokens
        tokens = [self.inv_vocab.get(id, self.unk_token) for id in takewhile(lambda x: x != self.pad_id, token_ids)]

        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_vocab]

        return " ".join(tokens)
    
class UnigramTokenizer:
    def __init__(self, max_length: int = 77):

        # get from encoders/get_checkpoints.py
        tokenizer_json_path = os.path.join(os.getcwd(), "encoders", "hub", "checkpoints", "t5_tokenizer", "tokenizer.json")

        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        vocab_list = data['model']['vocab']  # list of [piece, log-prob]

        # lookup tables
        self.token_scores = {piece: score for piece, score in vocab_list}
        self.token_to_id  = {piece: idx   for idx, (piece, _) in enumerate(vocab_list)}

        self.id_to_token  = {idx: piece for piece, idx in self.token_to_id.items()}
        self.max_length   = max_length

        # 3) Ensure an <unk> token exists
        self.unk_token = '<unk>'
        max_score = max(self.token_scores.values())

        if self.unk_token not in self.token_scores:

            unk_id = len(self.token_to_id)
            self.token_to_id[self.unk_token]  = unk_id

            self.token_scores[self.unk_token] = max_score
            self.id_to_token[unk_id]         = self.unk_token

        # 4) compute the longest piece ( '▁' counts as one char )
        self.max_piece_len = max(len(piece) for piece in self.token_scores)

    def encode(self, text: str):
        """ Searches for the best way of splitting text by maximizing their probabilities """

        # replace spaces with SentencePiece's underline
        text = text.replace(' ', '▁')
        N = len(text)

        # B) DP arrays: dp[i]=best score to cover text[:i], prev[i]=split point
        dp   = [float('inf')] * (N + 1)
        prev = [0] * (N + 1)
        dp[0] = 0

        # Viterbi forward pass
        for i in range(1, N + 1):
            j0 = max(0, i - self.max_piece_len)
            for j in range(j0, i):
                piece = text[j:i]
                if piece in self.token_scores:
                    sc = dp[j] + self.token_scores[piece]
                    if sc < dp[i]:
                        dp[i]   = sc
                        prev[i] = j
            # fallback to UNK if nothing matched
            if dp[i] == float('inf'):
                dp[i]   = dp[i-1] + self.token_scores[self.unk_token]
                prev[i] = i - 1

        # backtrace to recover pieces
        pieces = []
        i = N
        while i > 0:
            j = prev[i]
            piece = text[j:i]
            if piece not in self.token_scores:
                piece = self.unk_token
            pieces.append(piece)
            i = j

        pieces = pieces[::-1] # reverse list

        # map to IDs
        ids = [self.token_to_id.get(p, self.token_to_id[self.unk_token]) for p in pieces]
        attn = torch.ones(self.max_length, dtype = torch.long) # initalize attn mask

        # truncate
        if len(ids) >= self.max_length:
            ids = ids[:self.max_length] 
        else:
            attn[len(ids):] = 0 # build attn mask
            ids += [self.token_to_id[self.unk_token]] * (self.max_length - len(ids)) # pad

        return torch.tensor(ids, dtype=torch.long), attn

    def decode(self, token_ids):
        # convert IDs back to pieces
        pieces = [self.id_to_token.get(int(i), self.unk_token) for i in token_ids]
        #join, restore spaces
        return ''.join(pieces).replace('▁', ' ').strip()
    
    
def test_tokenizer():
    tokenizer = TorchTokenizer()
    text = "hello world"
    token_ids, _ = tokenizer.tokenize(text)

    tokenizer = UnigramTokenizer()
    token_ids, attn = tokenizer.encode(text)

    print("Tokenized:", token_ids) 
    print("Attention Mask: ", attn)
    print("Detokenized:", tokenizer.decode(token_ids.tolist()))

if __name__ == "__main__":
    test_tokenizer()
import os
import json
import torch
import regex as re
import sentencepiece as spm # required for T5FastTokenizer
from torch.nn.utils.rnn import pad_sequence # for dynamic padding

def to_utf_bytes():
    # ord returns the unicode of a char
    bytes = (
          list(range(ord("!"), ord("~") + 1))   # ASCII Characters 33-126
        + list(range(ord("¡"), ord("¬") + 1))   # Latin‑1 supplement: 161–172
        + list(range(ord("®"), ord("ÿ") + 1))   # Latin‑1 supplement: 174–255
    )

    # copy the bytes
    characters = bytes[:]
    n = 0

    # add unadded bytes by mapping them into high codepoints (256, 257, ...)
    for byte in range(256): # 8-bits
        if byte not in bytes:
            bytes.append(byte)
            characters.append(256 + n)
            n += 1

    # turn bytes to chars
    characters = [chr(n) for n in characters]

    return dict(zip(bytes, characters))

def get_pairs(word):
    " Turn words (e.g ['r', 'e', 'a', 'd'] into pairs like tuple('r', 'e'), tuple('e', 'a'), ... ) "

    prev_word = word[0]
    pairs = set()

    for word in word[1:]:
        pairs.add((prev_word, word))
        prev_word = word

    return pairs
    
class TorchTokenizer: # Byte-Level Byte-Pair Tokenizer
    def __init__(self, tokenizer_path: str = "tokenizer.json", max_length: int = 77):

        # Load the tokenizer.json file
        file_path = os.path.join(os.getcwd(), "src", "model", tokenizer_path)

        with open(file_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)
        
        self.max_length = max_length

        # extract vocabulary and merges
        self.vocab = tokenizer_data["model"]["vocab"]
        merges = tokenizer_data["model"]["merges"]

        # split merges on white space "re ad" -> tuple("re", "ad")
        bpe_merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        self.byte_encoder = to_utf_bytes()

        self.vocab_len = len(self.vocab)

        self.start = tokenizer_data['added_tokens'][0]['id']
        self.end = tokenizer_data['added_tokens'][1]['id']

        self.pad_id = self.unk_token = self.end

        # will later expand with added tokens from tokenization
        self.cache = { "<|startoftext|>": "<|startoftext|>",  "<|endoftext|>": "<|endoftext|>"}

        self.pattern = re.compile(
            #           special tokens                 contractions      get letters and digits  other non-whitespace
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE # ignore Capital Cases A = a in this pattern search
        )

    def bpe_tokenize(self, token):

        """ Applies the merging logic """
        
        # if token was previously tokenize, just return it from the cache
        if token in self.cache:
            return self.cache[token]

        # all the words in a tuple + last letter with end of word token
        word = tuple(token[:-1]) + tuple(token[-1] + "</w>")
        pairs = get_pairs(word)

        # if not pairs (single word)
        if pairs:
            return token + "</w>"
        
        # iterate over highest-priority pairs and tokenize them
        # until all the merges happen (text gets tokenized)
        while True:
            # map pairs to the most occuring merges according to their rank (higher rank -> merge earlier)
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float("inf")))

            # if the end has been reached (full text tokenization)
            if bigram not in self.bpe_ranks:
                break
            
            # unpack the two symbols you're about to merge
            first, second = bigram

            new_word = []
            i = 0

            # find the symbols and reconstruct the word
            while i < len(word):

                # get the index of the first word
                try:
                    j = word.index(first, i)
                # if there no more 'first' letter occuring, then it will throw a ValueError
                except ValueError:
                    # index was found (letter) in the word,
                    # so escape because there's nothing to merge
                    new_word.extend(word[i:])
                    break
                else: 
                    # if succeeded, copy the letters from i to j (where the first letter index appeared)
                    new_word.extend(word[i:j])
                    i = j
                
                if word[i] == first and word[i + 1] == second and i < len(word):
                    # if pair was found, append it and increment by 2
                    new_word.append(first + second)
                    i += 2
                    # continue until ValueError is raised

                else:
                    # if no pair was found, continue searching
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)
            word = new_word

            # if the vocab has a full subword, exit (word collabsed into a single symbol)
            # otherwise it will continue or exist if if bigram not in self.bpe_ranks
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        word = " ".join(word)
        self.cache[word] = word

        return word

    def tokenize(self, text):
        """Tokenizes text by splitting on whitespace and maps to vocab."""  

        # normalize
        text = text.lower()

        token_ids = []
        # use re to split text on whitespace and only get useful data
        for token in re.findall(self.pattern, text):
            # turn data to utf-8 bytes
            token = "".join(
                self.byte_encoder[t] for t in token.encode("utf-8")
            )

            bpe_tokens = self.bpe_tokenize(token).split(" ")

            for sub in bpe_tokens:
                token_ids.append(self.vocab.get(sub, self.unk_token))

        # add start and end tokens
        token_ids = [self.start] + token_ids + [self.end]

        # truncate
        if len(token_ids) < self.max_length:
            token_ids = token_ids[:self.max_length]

        return torch.tensor(token_ids, dtype = torch.long)
    
    def tokenize_batch(self, texts):

        " Dynamic padding "

        input_ids = [torch.tensor(self.tokenize(text), dtype = torch.long) for text in texts]

        padded_ids = pad_sequence(input_ids, batch_first = True, padding_value = self.pad_id)

        attention_mask = (padded_ids != self.pad_id).long()

        # get the eos token and turn it to one in the attention mask
        # this is beacause eos and pad token share the same id, so it gets padded out
        eos_id = attention_mask.argmax(dim = 1)
        attention_mask[eos_id] = 1

        return padded_ids, attention_mask
    
class UnigramTokenizer: # For T5 Encoder
    def __init__(self, max_length: int = 77):

        # get from encoders/get_checkpoints.py
        tokenizer_json_path = os.path.join(os.getcwd(), "encoders", "hub", "checkpoints", "t5_tokenizer", "spiece.model")

        self.spm_model = spm.SentencePieceProcessor()
        self.spm_model.Load(tokenizer_json_path)

        self.max_length = max_length
        
        # 3) Ensure an special token exists
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.eos_token = "</s>"

        self.unk_token_length = len(self.spm_model.encode(str(self.unk_token)))

    # convert a char piece to an integer
    def _piece_to_id(self, piece: str) -> int:
        return self.spm_model.piece_to_id(piece)
        
    def encode(self, text: str):
        """ Searches for the best way of splitting text by maximizing their probabilities """

        # replace spaces with SentencePiece's underline
        tokens = self.spm_model.encode(text, out_type = str)

        tokens = tokens + [self.eos_token]

        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        
        return tokens
    
    def encode_ids(self, text):
        tokens = self.encode(text)
        ids = [self._piece_to_id(token) for token in tokens]
        return torch.tensor(ids)
    
def test_tokenizer():
    
    text = "a photo of a cat"

    tokenizer = TorchTokenizer()
    token_ids = tokenizer.tokenize(text)

    tokenizer = UnigramTokenizer()
    token_ids = tokenizer.encode_ids(text)

    print("Tokenized:", token_ids) 

if __name__ == "__main__":
    test_tokenizer()
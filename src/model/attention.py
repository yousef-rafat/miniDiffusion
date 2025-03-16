import torch
import torch.nn as nn

class PagedAttention(nn.Module):
    def __init__(self, heads: int, embedding_size: int, dropout: float, page_size: int = 512, max_pages: int = 16):
        """
        Creates a Paged KV Cache Attention Mechanisim
        """
        super(PagedAttention, self).__init__()

        self.heads = heads
        self.embedding_size = embedding_size
        self.page_size = page_size
        self.max_pages = max_pages

        assert embedding_size % heads == 0, "embedding size must be divisible by heads"
        self.head_dim = embedding_size // heads

        self.dropout_layer = nn.Dropout(dropout)

        self.q_linear = nn.Linear(embedding_size, embedding_size)
        self.k_linear = nn.Linear(embedding_size, embedding_size)
        self.v_linear = nn.Linear(embedding_size, embedding_size)

        self.out_linear = nn.Linear(embedding_size, embedding_size)

        # KV Cache Storage (Paged)
        self.kv_cache = []  # Stores (K, V) pages

    # key and value equal None to compliy with PyTorch's api
    def forward(self, x: torch.Tensor, key = None, value = None, attn_mask = None, use_cache: bool = True, key_padding_mask: torch.Tensor = None, **kwargs):
        # reference attention equation
        # https://pbs.twimg.com/profile_images/1624054272676532224/UNv4ONME_400x400.jpg

        if key is None: key = x
        if value is None: value = x

        # handle padding tokens
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1) == 1, 0) 

        batch_size, seq_length, _ = x.size()

        Q = self.q_linear(x)
        new_K = self.k_linear(x)
        new_V = self.v_linear(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        new_K = new_K.view(batch_size, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        new_V = new_V.view(batch_size, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # store in cache and get the full cache
        if use_cache:
            self.store_in_cache(new_K, new_V)

            K, V = self.get_cached_kv()
        else:
            K, V = new_K, new_V 

        # compute attention, reference above
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # check if attention scores are valid
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))
        
        attention = torch.softmax(scores, dim = -1)

        attention = self.dropout_layer(attention)

        output = torch.matmul(attention, V)

        # (batch_size, seq_length, embedding_size)
        out = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.embedding_size)

        return self.out_linear(out)

    def store_in_cache(self, K, V):
        """Stores K and V in paged memory format."""
        for i in range(K.shape[2]):  # Iterate over sequence length
            token_K = K[:, :, i, :]  # Select token K across batch & heads
            token_V = V[:, :, i, :]  # Select token V

            if len(self.kv_cache) >= self.max_pages * self.page_size:
                # if cache exceeds max capacity, remove the oldest entry
                # FIFO
                self.kv_cache.pop(0)

            self.kv_cache.append((token_K, token_V))  # Store as a new entry

    def get_cached_kv(self):
        """Retrieves stored KV pairs and returns as tensors."""
        if not self.kv_cache:
            return None, None  # Empty cache case

        K_list, V_list = zip(*self.kv_cache)  # Unpack stored KV pairs
        K = torch.cat(K_list, dim=-2)  # Stack along sequence axis
        V = torch.cat(V_list, dim=-2)

        return K, V

    def reset_cache(self):
        """clears stored KV cache."""
        self.kv_cache = []

    def __del__(self):
        super()
        self.reset_cache()

class PagedTransformerEncoderLayer(nn.TransformerEncoderLayer):
    " Transformer Encoder Layer with Paged Attention "
    def __init__(self, embed_dim, num_heads, dim_feedforward: int = 2048, dropout = 0.1):
        # dim_feedforward = number of hidden features in FFN
        super().__init__(embed_dim, num_heads, dim_feedforward, dropout)
        self.self_attn = PagedAttention(heads = num_heads, embedding_size = embed_dim, dropout = dropout)

def test_atten(embed_dim = 512):
    x = torch.rand(1024, embed_dim)
    key_padding_mask = torch.zeros(1, 1024) 
    model = PagedTransformerEncoderLayer(embed_dim = embed_dim, num_heads = 8)
    output = model(x.unsqueeze(0), src_key_padding_mask = key_padding_mask)
    print(output)
    del model
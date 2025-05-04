import torch
import torch.nn as nn
from collections import deque
from dit_components import RMSNorm
class PagedJointAttention(nn.Module):
    def __init__(self, heads: int, embedding_size: int, dropout: float = 0.1, page_size: int = 512, max_pages: int = 16, 
                batch_size: int = 1):
        """
        Creates a Paged KV Cache Attention Mechanisim
        """
        super(PagedJointAttention, self).__init__()

        self.heads = heads
        self.embedding_size = embedding_size
        self.page_size = page_size
        self.max_pages = max_pages
        self.max_tokens = max_pages * page_size

        # normalization
        self.norm_q = RMSNorm(embedding_size)
        self.norm_k = RMSNorm(embedding_size)
        self.norm_added_q = RMSNorm(embedding_size)
        self.norm_added_k = RMSNorm(embedding_size)

        self.to_out = nn.ModuleList([
            nn.Linear(embedding_size, embedding_size),
            nn.Dropout(dropout)
        ])

        # avoid error
        self.batch_first = False

        assert embedding_size % heads == 0, "embedding size must be divisible by heads"
        self.head_dim = embedding_size // heads

        self.to_q = nn.Linear(embedding_size, embedding_size)
        self.to_k = nn.Linear(embedding_size, embedding_size)
        self.to_v = nn.Linear(embedding_size, embedding_size)

        # for joint attention
        self.add_q_proj = nn.Linear(embedding_size, embedding_size)
        self.add_k_proj = nn.Linear(embedding_size, embedding_size)
        self.add_v_proj = nn.Linear(embedding_size, embedding_size)

        self.to_add_out = nn.Linear(embedding_size, embedding_size)

        self.k_cache = deque(maxlen = self.max_pages)
        self.v_cache = deque(maxlen = self.max_pages)

    # key and value equal None to compliy with PyTorch's api
    def forward(self, x: torch.Tensor, key = None, value = None, attn_mask = None, use_cache: bool = True, key_padding_mask: torch.Tensor = None, 
                encoder_hidden_state: torch.Tensor = None, **kwargs):
        # reference attention equation
        # https://pbs.twimg.com/profile_images/1624054272676532224/UNv4ONME_400x400.jpg

        if key is None: key = x
        if value is None: value = x

        residual = x

        # handle padding tokens
        if key_padding_mask is not None:
            # handle cases where broadcasting is and isn't needed
            try: x = x.masked_fill(key_padding_mask.unsqueeze(-1) == 1, 0).float()
            except RuntimeError:
                x = x.masked_fill(key_padding_mask == 1, 0).float()

        batch_size, seq_length, _ = x.size()

        Q = self.to_q(x)
        new_K = self.to_k(x)
        new_V = self.to_v(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        new_K = new_K.view(batch_size, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        new_V = new_V.view(batch_size, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # store in cache and get the full cache
        if use_cache:
            self.store_in_cache(new_K, new_V)

            # get the previous K and V for complete context
            K, V = self.get_cached_kv()
        else:
            K, V = new_K, new_V 

        # normalize query and key
        print(Q.size())
        Q = self.norm_q(Q)
        K = self.norm_k(K)

        # the Joint Attention part
        if encoder_hidden_state is not None:
            
            # forward pass with reshaping 
            encoder_query = self.add_q_proj(encoder_hidden_state)\
                .view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2) # (B, seq_len, heads, heads_dim)
            
            encoder_key = self.add_k_proj(encoder_hidden_state)\
                .view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2) # //
            
            encoder_value = self.add_v_proj(encoder_hidden_state)\
                .view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2) # //
            
            # normalize
            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            # concat the attentions
            Q = torch.concat([Q, encoder_query], dim = 2)
            K = torch.concat([K, encoder_key], dim = 2)
            V = torch.concat([V, encoder_value], dim = 2)

        # compute attention, reference above
        try: scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        except:
            # infer correct seq_len after returning from cache
            seq_length = K.shape[1] // self.heads

            # reshape to correct sizes
            K = K.view(batch_size, self.heads, seq_length, self.head_dim)
            V = V.view(batch_size, self.heads, seq_length, self.head_dim)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # check if attention scores are valid
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))
        
        attention = torch.softmax(scores, dim = -1)

        #attention = self.dropout_layer(attention)
        output = torch.matmul(attention, V)

        # (batch_size, seq_length, embedding_size)
        seq_length = output.size(2)
        out = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.embedding_size)

        # split hidden_states and encoder hidden states
        if encoder_hidden_state is not None:
            hidden_states, encoder_hidden_states = (
                out[:, :residual.size(1)],
                out[:, residual.size(1):]
            )

            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        # linear + dropout
        for layer in self.to_out:
            hidden_states = layer(hidden_states)

        if encoder_hidden_state is None: return hidden_states

        else: return hidden_states, encoder_hidden_states

    def store_in_cache(self, K_new: torch.Tensor, V_new: torch.Tensor):
        """
        Splits incoming K/V by pages of size `page_size` and appends each chunk to deques.
        """

        k_chunks = torch.split(K_new, self.page_size, dim = 2)
        v_chunks = torch.split(V_new, self.page_size, dim = 2)

        for k_chunk, v_chunk in zip(k_chunks, v_chunks):
            self.k_cache.append(k_chunk)
            self.v_cache.append(v_chunk)

    def get_cached_kv(self):
        """
        Concatenate all cached pages along sequence dim to form full K and V.
        """
        if not self.k_cache:
            return None, None
        K = torch.cat(list(self.k_cache), dim=2)
        V = torch.cat(list(self.v_cache), dim=2)
        return K, V

    def reset_cache(self):
        """Clear all cached pages."""
        self.k_cache.clear()
        self.v_cache.clear()

    def __del__(self):
        super()
        self.reset_cache()

class PagedTransformerEncoderLayer(nn.TransformerEncoderLayer):
    " Transformer Encoder Layer with Paged Attention "
    def __init__(self, embed_dim, num_heads, dim_feedforward: int = 2048, dropout = 0.1):
        # dim_feedforward = number of hidden features in FFN
        super().__init__(embed_dim, num_heads, dim_feedforward, dropout)
        self.self_attn = PagedJointAttention(heads = num_heads, embedding_size = embed_dim, dropout = dropout)

def test_atten(embed_dim = 512):
    x = torch.rand(1024, embed_dim)
    key_padding_mask = torch.zeros(1, 1024) 
    model = PagedTransformerEncoderLayer(embed_dim = embed_dim, num_heads = 8)
    output = model(x.unsqueeze(0), src_key_padding_mask = key_padding_mask)
    print(output)
    del model

if __name__ == "__main__":
    test_atten()
import os
import json
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn


def create_causal_mask(n: int):
    return torch.tril(torch.ones(n, n), diagonal=0).type(torch.uint8)


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_len: int = 512
    dropout: float = 0.1


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v):
        """
        q - (batch_size, n_head, seq_len, head_dim)
        k - (batch_size, n_head, seq_len, head_dim)
        v - (batch_size, n_head, seq_len, head_dim)
        """
        d_k = k.size(-1)
        seq_len = k.size(-2)
        attn_weights = q @ k.transpose(-1, -2) / d_k ** 0.5  # (batch_size, n_head, seq_len, seq_len)

        # Create mask inside sdpa because attention is always
        # used with causal mask in decoder-only transformer
        mask = create_causal_mask(seq_len)  # (seq_len, seq_len)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        attn_weights = torch.masked_fill(attn_weights, mask == 0, float('-inf'))
        attn_probabilities = attn_weights.softmax(dim=-1)
        return attn_probabilities @ v  # (batch_size, n_head, seq_len, head_dim)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout=0.1):
        super().__init__()

        assert hidden_size % num_attention_heads == 0
        self.n_head = num_attention_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_attention_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
    
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.sdpa = ScaledDotProductAttention(dropout)

    def forward(self, x):
        """
        x - (batch_size, seq_len, hidden_size)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        sdpa = self.sdpa(q, k, v)  # (batch_size, n_head, seq_len, head_dim)
        sdpa = sdpa.transpose(1, 2).contiguos().view(batch_size, seq_len, self.hidden_size)
        return self.proj(sdpa)  # (batch_size, seq_len, hidden_size)


# https://arxiv.org/pdf/1606.08415
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                (2 / torch.pi) ** 0.5 *
                (x + 0.044715 * x ** 3)
            )
        )


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )

    def forward(self, x):
        return self.ffn(x)  # (batch_size, seq_len, hidden_size)


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(hidden_size, num_attention_heads, dropout)
        self.ffn = FeedForwardNetwork(hidden_size, intermediate_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
    
    def forward(self, x):
        attention = self.layer_norm1(x + self.dropout1(self.mha(x)))
        return self.layer_norm2(attention + self.dropout2(self.ffn(attention)))  # (batch_size, seq_len, hidden_size)


class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_attention_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)
        return x  # (batch_size, seq_len, hidden_size)


class GPT(nn.Module):
    def __init__(
        self,
        config: GPTConfig,
    ):
        super().__init__()
        self.config = config
        self.sqrt_model = config.hidden_size ** 0.5
        self.decoder = Decoder(config.hidden_size, config.num_layers, config.num_attention_heads, config.intermediate_size, config.dropout)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_embeddings = nn.Embedding(config.max_len, config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(p=config.dropout)

        self.proj.weight = self.embeddings.weight

    def forward(self, x, return_embeddings=False):
        """
        x - (batch_size, seq_len)
        """
        seq_len = x.size(1)
        embeddings = self.embeddings(x) * self.sqrt_model
        pos_embeddings = self.positional_embeddings(torch.arange(seq_len))  # (seq_len, hidden_size)
        embeddings += pos_embeddings.unsqueeze(0)
        embeddings = self.dropout(embeddings)
        x = self.decoder(x)  # (batch_size, seq_len, hidden_size)
        if return_embeddings:
            return x
        return self.proj(x)  # (batch_size, seq_len, vocab_size)

    def save_pretrained(self, save_path: str) -> None:
        torch.save(self.state_dict(), os.path.join(save_path, "model.pt"))
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_path: str):
        with open(os.path.join(pretrained_path, "config.json")) as f:
            config = json.load(f)
        model = cls(GPTConfig(**config))
        state_dict = torch.load(os.path.join(pretrained_path, "model.pt"))
        model.load_state_dict(state_dict)
        return model

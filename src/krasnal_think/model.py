"""Decoder-only transformer: RoPE, RMSNorm, SwiGLU, SDPA, incremental KV cache.

Derived from nanoGPT. SDPA rather than flash-attn so the same code runs on CUDA,
MPS and CPU; on CUDA it dispatches to the flash kernel anyway.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 2112
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    rope_theta: float = 10000.0


class KVCache:
    """Preallocated per-layer key/value store for single-sequence decoding."""

    def __init__(self, config: GPTConfig, batch: int, device, dtype):
        head_dim = config.n_embd // config.n_head
        shape = (config.n_layer, batch, config.n_head, config.block_size, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        end = self.length + k.size(2)
        if end > self.k.size(3):
            raise IndexError(f"cache overflow: {end} > block_size {self.k.size(3)}")
        self.k[layer, :, :, self.length : end] = k
        self.v[layer, :, :, self.length : end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def advance(self, n: int):
        self.length += n

    def truncate(self, length: int):
        self.length = min(self.length, length)


class Attention(nn.Module):
    def __init__(self, config: GPTConfig, layer: int):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head, self.n_embd, self.layer = config.n_head, config.n_embd, layer
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        head_dim = config.n_embd // config.n_head
        freqs = config.rope_theta ** -((torch.arange(head_dim) // 2).float() / (head_dim // 2))
        phases = torch.arange(config.block_size).float()[:, None] * freqs[None, :]
        self.rope_cos = nn.Buffer(torch.cos(phases), persistent=False)
        self.rope_sin = nn.Buffer(torch.sin(phases), persistent=False)

    def _rope(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        t = x.size(2)
        cos = self.rope_cos[offset : offset + t].to(x)
        sin = self.rope_sin[offset : offset + t].to(x)
        rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).view_as(x)
        return x * cos + rotated * sin

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = (t.view(B, T, self.n_head, -1).transpose(1, 2) for t in (q, k, v))
        offset = cache.length if cache is not None else 0
        q, k = self._rope(q, offset), self._rope(k, offset)
        if cache is not None:
            k, v = cache.update(self.layer, k, v)
        # SDPA's is_causal aligns top-left, which is only correct when the query
        # block starts at position 0; otherwise mask explicitly against the cache.
        mask = None
        if offset > 0 and T > 1:
            positions = torch.arange(T, device=x.device) + offset
            mask = positions[:, None] >= torch.arange(k.size(2), device=x.device)[None, :]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=mask is None and T > 1,
        )
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = (8 * config.n_embd + 2) // 3
        self.c_fc = nn.Linear(config.n_embd, 2 * hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.c_fc(x).chunk(2, dim=-1)
        return self.c_proj(value * F.silu(gate))


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer: int):
        super().__init__()
        self.norm_1 = nn.RMSNorm(config.n_embd)
        self.attn = Attention(config, layer)
        self.norm_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm_1(x), cache))
        return x + self.drop(self.mlp(self.norm_2(x)))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config, i) for i in range(config.n_layer))
        self.norm_f = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, cache: KVCache | None = None):
        x = self.drop(self.wte(idx))
        for block in self.blocks:
            x = block(x, cache)
        if cache is not None:
            cache.advance(idx.size(1))
        x = self.norm_f(x)
        if targets is None:
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-1)
        return logits, loss

    @classmethod
    def load(cls, path, device: str = "cpu") -> "GPT":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(GPTConfig(**checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["model"])
        return model.eval()

    def configure_optimizers(self, weight_decay, learning_rate, betas, fused=False):
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=fused)

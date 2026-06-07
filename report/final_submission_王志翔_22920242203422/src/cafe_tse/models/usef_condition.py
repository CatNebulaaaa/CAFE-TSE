from __future__ import annotations

import torch
import torch.nn as nn


class USEFConditionExtractor(nn.Module):
    """Embedding-free USEF-style target condition extractor."""

    def __init__(self, dim: int, n_heads: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, mix_tokens: torch.Tensor, enroll_tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(query=mix_tokens, key=enroll_tokens, value=enroll_tokens)
        attended = self.norm(attended)
        return attended.mean(dim=1)

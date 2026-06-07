from __future__ import annotations

import torch
import torch.nn as nn

from cafe_tse.models.sparse_fusion import SparseConditionFusion


class GridLiteBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, n_heads: int):
        super().__init__()
        self.channel_mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
        self.freq_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F, C]
        x = x + self.channel_mlp(x)
        b, t, f, c = x.shape
        xf = x.reshape(b * t, f, c).transpose(1, 2)
        xf = self.freq_conv(xf).transpose(1, 2).reshape(b, t, f, c)
        x = x + xf
        xt = x.permute(0, 2, 3, 1).reshape(b * f, c, t)
        xt = self.temporal_conv(xt).reshape(b, f, c, t).permute(0, 3, 1, 2)
        x = x + xt
        pooled = x.mean(dim=2)
        attended, _ = self.attn(pooled, pooled, pooled)
        x = x + attended.unsqueeze(2)
        return self.norm(x)


class TFGridNetLite(nn.Module):
    """Lightweight time-frequency grid separator with dynamic active blocks."""

    def __init__(
        self,
        emb_dim: int = 32,
        hidden_dim: int = 128,
        n_blocks: int = 4,
        n_heads: int = 2,
        sparse_fusion_blocks: list[int] | None = None,
        condition_fusion_mode: str = "film",
        output_mode: str = "mag_mask",
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.sparse_fusion_blocks = list(sparse_fusion_blocks or [])
        self.output_mode = output_mode
        self.input_proj = nn.Linear(2, emb_dim)
        self.blocks = nn.ModuleList([GridLiteBlock(emb_dim, hidden_dim, n_heads) for _ in range(n_blocks)])
        self.fusions = nn.ModuleDict(
            {str(i): SparseConditionFusion(emb_dim, emb_dim, mode=condition_fusion_mode) for i in self.sparse_fusion_blocks}
        )
        self.output_proj = nn.Linear(emb_dim, 2)

    def forward(
        self,
        spec: torch.Tensor,
        mix_tokens: torch.Tensor,
        condition: torch.Tensor,
        active_blocks: list[int] | None = None,
        fusion_blocks_by_sample: list[list[int]] | None = None,
    ) -> torch.Tensor:
        del mix_tokens
        b, f, t = spec.shape
        ri = torch.stack([spec.real, spec.imag], dim=-1).transpose(1, 2)  # [B, T, F, 2]
        x = self.input_proj(ri)
        if active_blocks is None:
            active_blocks = [self.n_blocks] * b
        active = torch.tensor(active_blocks, device=x.device).view(b, 1, 1, 1)

        for idx, block in enumerate(self.blocks):
            mask = (active > idx).to(dtype=x.dtype)
            if not bool(mask.any()):
                break
            candidate = x
            if str(idx) in self.fusions:
                fused = self.fusions[str(idx)](candidate, condition)
                if fusion_blocks_by_sample is None:
                    candidate = fused
                else:
                    fusion_mask = torch.tensor(
                        [idx in blocks for blocks in fusion_blocks_by_sample],
                        device=x.device,
                        dtype=x.dtype,
                    ).view(b, 1, 1, 1)
                    candidate = fused * fusion_mask + candidate * (1.0 - fusion_mask)
            candidate = block(candidate)
            x = candidate * mask + x * (1.0 - mask)

        pred = self.output_proj(x).transpose(1, 2)  # [B, F, T, 2]
        if self.output_mode == "mag_mask":
            mask = torch.sigmoid(pred[..., 0])
            return spec * mask
        if self.output_mode == "complex_mask":
            mask = torch.complex(torch.tanh(pred[..., 0]), torch.tanh(pred[..., 1]))
            return spec * mask
        residual = 0.1 * torch.tanh(pred)
        res_complex = torch.complex(residual[..., 0], residual[..., 1])
        return spec + res_complex

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from cafe_tse.features.complexity import compute_complexity_score
from cafe_tse.models.dynamic_router import DynamicRouter
from cafe_tse.models.tfgridnet_lite import TFGridNetLite
from cafe_tse.models.usef_condition import USEFConditionExtractor


@dataclass
class CafeTSEOutput:
    wav: torch.Tensor
    route: list[str]
    complexity_score: torch.Tensor
    active_blocks: list[int]
    rtf: float | None = None


class CafeTSE(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int | None = None,
        emb_dim: int = 32,
        hidden_dim: int = 128,
        n_blocks: int = 4,
        n_heads: int = 2,
        sparse_fusion_blocks: list[int] | None = None,
        dynamic_inference: bool = False,
        shallow_blocks: int = 2,
        lite_blocks: int = 3,
        threshold_easy: float = 0.35,
        threshold_hard: float = 0.65,
        egsp_enabled: bool = False,
        egsp_strength: float = 0.25,
        egsp_min_weight: float = 0.70,
        egsp_max_weight: float = 1.30,
        egsp_apply_to_spec: bool = False,
        condition_fusion_mode: str = "film",
        dynamic_sparse_fusion: bool = False,
        separator_output_mode: str = "mag_mask",
        reference_anchor_weight: float = 0.0,
        **_: object,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.dynamic_inference = dynamic_inference
        self.shallow_blocks = shallow_blocks
        self.lite_blocks = lite_blocks
        self.full_blocks = n_blocks
        self.egsp_enabled = egsp_enabled
        self.egsp_strength = egsp_strength
        self.egsp_min_weight = egsp_min_weight
        self.egsp_max_weight = egsp_max_weight
        self.egsp_apply_to_spec = egsp_apply_to_spec
        self.dynamic_sparse_fusion = dynamic_sparse_fusion
        self.reference_anchor_weight = float(reference_anchor_weight)
        self.sparse_fusion_blocks = list(range(n_blocks)) if sparse_fusion_blocks is None else list(sparse_fusion_blocks)
        self.mix_encoder = nn.Linear(n_fft // 2 + 1, emb_dim)
        self.enroll_encoder = nn.Linear(n_fft // 2 + 1, emb_dim)
        self.condition = USEFConditionExtractor(emb_dim, n_heads)
        self.separator = TFGridNetLite(
            emb_dim,
            hidden_dim,
            n_blocks,
            n_heads,
            self.sparse_fusion_blocks,
            condition_fusion_mode=condition_fusion_mode,
            output_mode=separator_output_mode,
        )
        self.router = DynamicRouter(threshold_easy, threshold_hard)

    def _stft_mag(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        window = torch.hann_window(self.win_length, device=wav.device, dtype=wav.dtype)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        return spec, spec.abs()

    def _egsp_weight(self, enroll_mag: torch.Tensor) -> torch.Tensor:
        profile = enroll_mag.mean(dim=-1)
        profile = profile / profile.mean(dim=1, keepdim=True).clamp_min(1e-6)
        weight = 1.0 + self.egsp_strength * (profile - 1.0)
        return weight.clamp(self.egsp_min_weight, self.egsp_max_weight)

    def _fusion_blocks_by_complexity(self, complexity: torch.Tensor) -> list[list[int]]:
        if not self.sparse_fusion_blocks:
            return [[] for _ in range(complexity.shape[0])]
        routes = self.router(complexity)
        ordered = self.sparse_fusion_blocks
        out = []
        for route in routes:
            if route == "shallow":
                out.append(ordered[:1])
            elif route == "lite":
                out.append(ordered[: max(1, min(2, len(ordered)))])
            else:
                out.append(ordered)
        return out

    def forward(
        self,
        mixture: torch.Tensor,
        enrollment: torch.Tensor,
        active_blocks_override: list[int] | None = None,
    ) -> CafeTSEOutput:
        start = time.perf_counter() if not self.training else None
        mix_spec, mix_mag = self._stft_mag(mixture)
        _, enroll_mag = self._stft_mag(enrollment)
        separator_spec = mix_spec
        if self.egsp_enabled:
            egsp_weight = self._egsp_weight(enroll_mag)
            mix_mag = mix_mag * egsp_weight.unsqueeze(-1)
            if self.egsp_apply_to_spec:
                separator_spec = mix_spec * egsp_weight.unsqueeze(-1)
        mix_tokens = self.mix_encoder(mix_mag.transpose(1, 2))
        enroll_tokens = self.enroll_encoder(enroll_mag.transpose(1, 2))
        cond = self.condition(mix_tokens, enroll_tokens)
        complexity = compute_complexity_score(mixture, self.sample_rate, self.n_fft, self.hop_length)

        if active_blocks_override is not None:
            routes = ["depth_aware"] * mixture.shape[0]
            active_blocks = active_blocks_override
        elif self.dynamic_inference and not self.training:
            routes = self.router(complexity)
            active_blocks = self.router.active_blocks(routes, self.shallow_blocks, self.lite_blocks, self.full_blocks)
        else:
            routes = ["full"] * mixture.shape[0]
            active_blocks = [self.full_blocks] * mixture.shape[0]

        fusion_blocks_by_sample = self._fusion_blocks_by_complexity(complexity) if self.dynamic_sparse_fusion else None
        est_spec = self.separator(
            separator_spec,
            mix_tokens,
            cond,
            active_blocks=active_blocks,
            fusion_blocks_by_sample=fusion_blocks_by_sample,
        )
        window = torch.hann_window(self.win_length, device=mixture.device, dtype=mixture.dtype)
        wav = torch.istft(
            est_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=mixture.shape[-1],
        )
        if self.reference_anchor_weight:
            weight = max(0.0, min(1.0, self.reference_anchor_weight))
            wav = (1.0 - weight) * wav + weight * enrollment[..., : wav.shape[-1]]
        rtf = None
        if start is not None:
            elapsed = time.perf_counter() - start
            audio_dur = mixture.shape[-1] / float(self.sample_rate)
            rtf = elapsed / max(audio_dur, 1e-8)
        return CafeTSEOutput(wav=wav, route=routes, complexity_score=complexity, active_blocks=active_blocks, rtf=rtf)


def build_model(cfg: dict) -> CafeTSE:
    model_cfg = dict(cfg.get("model", {}))
    model_cfg["sample_rate"] = cfg.get("sample_rate", model_cfg.get("sample_rate", 16000))
    model_cfg["threshold_easy"] = model_cfg.pop("route_threshold_easy", model_cfg.get("threshold_easy", 0.35))
    model_cfg["threshold_hard"] = model_cfg.pop("route_threshold_hard", model_cfg.get("threshold_hard", 0.65))
    return CafeTSE(**model_cfg)

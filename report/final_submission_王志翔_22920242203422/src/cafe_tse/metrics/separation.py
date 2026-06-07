from __future__ import annotations

import torch

from cafe_tse.losses.sisdr import si_sdr


def _optional_perceptual_metrics(est: torch.Tensor, target: torch.Tensor, sample_rate: int) -> dict[str, float]:
    import numpy as np

    n = min(est.numel(), target.numel())
    est_np = np.ascontiguousarray(est[:n].detach().float().cpu().numpy(), dtype=np.float64)
    target_np = np.ascontiguousarray(target[:n].detach().float().cpu().numpy(), dtype=np.float64)
    est_np = np.clip(est_np, -1.0, 1.0)
    target_np = np.clip(target_np, -1.0, 1.0)
    out = {"stoi": float("nan"), "pesq": float("nan")}
    try:
        from pystoi import stoi

        out["stoi"] = float(stoi(target_np, est_np, sample_rate, extended=False))
    except Exception:
        pass
    try:
        from pesq import pesq

        mode = "nb" if sample_rate <= 8000 else "wb"
        out["pesq"] = float(pesq(sample_rate, target_np, est_np, mode))
    except Exception:
        pass
    return out


def compute_bss_metrics(
    est: torch.Tensor,
    target: torch.Tensor,
    interferer: torch.Tensor,
    mixture: torch.Tensor,
    sample_rate: int = 16000,
) -> dict[str, float]:
    base = compute_basic_metrics(est, target, mixture, sample_rate)
    n = min(est.numel(), target.numel(), interferer.numel(), mixture.numel())
    est = est[:n].detach().float().cpu()
    target = target[:n].detach().float().cpu()
    interferer = interferer[:n].detach().float().cpu()
    refs = torch.stack([target, interferer], dim=1)
    gram = refs.T @ refs
    rhs = refs.T @ est
    eye = torch.eye(2, dtype=gram.dtype) * 1e-8
    coeffs = torch.linalg.solve(gram + eye, rhs)
    target_proj = coeffs[0] * target
    interf_proj = coeffs[1] * interferer
    artifact = est - target_proj - interf_proj

    def ratio_db(num: torch.Tensor, den: torch.Tensor) -> float:
        nrg_num = torch.sum(num**2).clamp_min(1e-12)
        nrg_den = torch.sum(den**2).clamp_min(1e-12)
        return float((10.0 * torch.log10(nrg_num / nrg_den)).item())

    base["sdr"] = ratio_db(target_proj, interf_proj + artifact)
    base["sir"] = ratio_db(target_proj, interf_proj)
    base["sar"] = ratio_db(target_proj + interf_proj, artifact)
    return base


def compute_basic_metrics(est: torch.Tensor, target: torch.Tensor, mixture: torch.Tensor, sample_rate: int = 16000) -> dict[str, float]:
    n = min(est.numel(), target.numel(), mixture.numel())
    est = est[:n].detach().float().cpu()
    target = target[:n].detach().float().cpu()
    mixture = mixture[:n].detach().float().cpu()
    est_score = si_sdr(est.unsqueeze(0), target.unsqueeze(0)).item()
    mix_score = si_sdr(mixture.unsqueeze(0), target.unsqueeze(0)).item()
    metrics = {
        "si_sdr": est_score,
        "si_sdri": est_score - mix_score,
        "sdr": est_score,
        "sir": est_score,
        "sar": est_score,
    }
    metrics.update(_optional_perceptual_metrics(est, target, sample_rate))
    return metrics

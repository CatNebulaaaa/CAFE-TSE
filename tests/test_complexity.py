import torch

from cafe_tse.features.complexity import compute_complexity_score, spectral_entropy


def test_complexity_score_range():
    wav = torch.randn(2, 1024)
    score = compute_complexity_score(wav, sample_rate=8000, n_fft=128, hop_length=32)
    assert score.shape == (2,)
    assert torch.all(score >= 0)
    assert torch.all(score <= 1)


def test_spectral_entropy_shape():
    mag = torch.rand(3, 65, 10)
    assert spectral_entropy(mag).shape == (3,)


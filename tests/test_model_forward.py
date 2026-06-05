import torch

from cafe_tse.models.cafe_tse import CafeTSE


def test_model_forward_shape():
    model = CafeTSE(sample_rate=8000, n_fft=128, hop_length=32, emb_dim=8, hidden_dim=16, n_blocks=2, n_heads=1, sparse_fusion_blocks=[0])
    mixture = torch.randn(2, 1024)
    enrollment = torch.randn(2, 1024)
    out = model(mixture, enrollment)
    assert out.wav.shape == mixture.shape
    assert len(out.route) == 2
    assert len(out.active_blocks) == 2


def test_model_defaults_are_tse_safe():
    model = CafeTSE(sample_rate=8000, n_fft=128, hop_length=32, emb_dim=8, hidden_dim=16, n_blocks=2, n_heads=1)
    assert model.separator.output_mode == "mag_mask"
    assert model.sparse_fusion_blocks == [0, 1]
    assert model.dynamic_inference is False

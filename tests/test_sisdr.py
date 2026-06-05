import torch

from cafe_tse.losses.sisdr import si_sdr


def test_sisdr_better_for_matching_signal():
    target = torch.randn(2, 1024)
    good = target.clone()
    bad = torch.randn(2, 1024)
    assert torch.all(si_sdr(good, target) > si_sdr(bad, target))


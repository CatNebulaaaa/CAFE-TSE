from pathlib import Path

from cafe_tse.cli.prepare_toy_data import main as toy_main
from cafe_tse.cli.prepare_librimix_manifest import _speaker_id_from_source
from cafe_tse.datasets.tse_dataset import TSEDataset


def test_dataset_loads_toy(tmp_path, monkeypatch):
    out = tmp_path / "toy"
    monkeypatch.setattr("sys.argv", ["prepare_toy_data", "--out_dir", str(out), "--num_samples", "2", "--sample_rate", "8000", "--duration", "0.5"])
    toy_main()
    ds = TSEDataset(out / "toy_manifest.csv", sample_rate=8000, segment_seconds=0.5)
    item = ds[0]
    assert item["mixture"].shape[-1] == 4000
    assert Path(item["utt_id"]).name.startswith("toy_")


def test_librimix_source_speaker_id_uses_source_index():
    stem = "100-121669-0026_718-129597-0003"
    assert _speaker_id_from_source(stem, 0) == "100"
    assert _speaker_id_from_source(stem, 1) == "718"

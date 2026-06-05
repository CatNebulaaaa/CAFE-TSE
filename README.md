# CAFE-TSE

CAFE-TSE is a PyTorch implementation of Curriculum-Aware Frequency-Efficient Target Speaker Extraction for the cocktail-party course project.

It supports:

- LibriMix / Libri2Mix manifest construction.
- Target-speaker enrollment references.
- Complexity scoring and curriculum difficulty labels.
- USEF-style target conditioning.
- TF-GridNet-Lite time-frequency separator.
- Sparse condition fusion and dynamic inference.
- SI-SDRi, SDR/SIR/SAR fallback metrics, RTF, params, active blocks, and skip ratio.
- Smoke and toy harnesses for local verification.

## Quick Smoke Test

```bash
# Install GPU PyTorch first according to the AutoDL CUDA version, for example:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
bash scripts/harness_smoke.sh
```

The smoke harness creates toy audio, computes complexity features, trains for one epoch, evaluates, runs inference, and executes tests.

On Windows PowerShell, run the equivalent commands:

```powershell
$env:PYTHONPATH="src"
python -m cafe_tse.cli.prepare_toy_data --out_dir data/toy --num_samples 8 --sample_rate 8000 --duration 1.0
python -m cafe_tse.cli.compute_complexity_manifest --manifest data/toy/toy_manifest.csv --out_manifest data/toy/toy_manifest_complexity.csv --sample_rate 8000
python -m cafe_tse.cli.train --config configs/smoke_tiny.yaml --train_manifest data/toy/toy_manifest_complexity.csv --valid_manifest data/toy/toy_manifest_complexity.csv --exp_dir experiments/smoke_tiny
python -m cafe_tse.cli.evaluate --config configs/smoke_tiny.yaml --checkpoint experiments/smoke_tiny/checkpoints/best.pt --test_manifest data/toy/toy_manifest_complexity.csv --out_dir results/smoke --save_audio 2
python -m cafe_tse.cli.infer --config configs/smoke_tiny.yaml --checkpoint experiments/smoke_tiny/checkpoints/best.pt --mixture data/toy/mixtures/toy_000.wav --enrollment data/toy/enrollments/toy_000.wav --out_wav results/smoke/audio/estimated_0.wav --device cpu
pytest tests/test_*.py -q
```

## LibriMix Small Workflow

```bash
python -m cafe_tse.cli.prepare_librimix_manifest \
  --librimix_root data/raw/LibriMix/Libri2Mix/wav16k/min \
  --out_dir data/metadata/librimix \
  --sample_rate 16000 \
  --num_speakers 2 \
  --mixture_type mix_clean \
  --max_train_samples 2000 \
  --max_valid_samples 200 \
  --max_test_samples 200

python -m cafe_tse.cli.compute_complexity_manifest \
  --manifest data/metadata/librimix/train_manifest.csv \
  --out_manifest data/metadata/librimix/train_manifest_final.csv \
  --sample_rate 16000

python -m cafe_tse.cli.train \
  --config configs/cafe_tse_dynamic.yaml \
  --train_manifest data/metadata/librimix/train_manifest_final.csv \
  --valid_manifest data/metadata/librimix/valid_manifest_final.csv \
  --exp_dir experiments/exp04_cafe_tse_dynamic
```

## Project Layout

Source code lives under `src/cafe_tse/`. CLI entry points are run with `python -m cafe_tse.cli.<name>`.

## GUI Prototype

Open `gui/index.html` in a browser to view the CAFE-TSE studio prototype. It provides case switching, waveform-style comparison, playback for the three demo audio cases, the final 2-speaker metrics, and the 3-speaker stress-test summary.

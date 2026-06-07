# TD-SpeakerBeam Target Speaker Extraction — Course Project

## Overview

This is the final submission for the "Cognitive and Computing" course project.
The report covers three tasks:

1. **Task 1**: Multi-dimensional comparison of AI and human intelligence
2. **Task 2**: Cocktail party target speaker extraction with TD-SpeakerBeam
   - Shared-clean80 training and evaluation (strong teacher 12.52 dB, mid student 10.43 dB, fine-tune distill 10.59 dB, multi-enrollment pooling 10.80 dB)
   - Enrollment sanity checks (correct/shuffled/interferer/zero/short/noisy)
   - Mixture SNR and 3-speaker stress tests
   - EGSP frequency pre-emphasis diagnostics
   - **Section 2.9**: Additional challenge experiments (streaming demo, DEMAND real noise, Chinese-English cross-lingual test, interactive Web GUI)
3. **Task 3**: Mapping neural networks and backpropagation to human learning mechanisms

## Additional Challenge Experiments (Section 2.9)

| Experiment | Script | Key Result |
|---|---|---|
| Streaming demo | `scripts/streaming_demo.py` | RTF 0.116 (8.6x real-time) on RTX 4080 SUPER |
| Real noise test | `scripts/test_real_noise.py` | DEMAND cafeteria noise: monotonic SI-SDR degradation with SNR |
| Cross-lingual test | `scripts/test_cross_lingual.py` | Both zh→en and en→zh directions failed (SI-SDR < -30 dB) |

Data sources (CC BY 4.0 / Apache 2.0): `results/data_sources.md`

## GUI Demo

Open `gui/index.html` in a browser to explore:
- System overview with architecture diagram
- 3 interactive audio cases with A/B comparison
- Radar and bar charts for experiment metrics
- Cognitive mapping of selective attention to model components

All demo audio files are in `demo_audio/` (from shared-clean80 test set actual outputs).

## Project Layout

```
final_submission/
├── report/
│   ├── 认知与计算课程报告_王志翔_22920242203422.pdf   (compiled, 29 pages)
│   └── 认知与计算课程报告_王志翔_22920242203422.tex   (source)
├── scripts/
│   ├── streaming_demo.py          (streaming chunk-by-chunk inference)
│   ├── test_real_noise.py         (real noise robustness test)
│   ├── test_cross_lingual.py      (cross-lingual test)
│   ├── train_open_speakerbeam.py  (TD-SpeakerBeam training)
│   ├── evaluate_open_speakerbeam_variants.py  (enrollment variants evaluation)
│   ├── evaluate_open_speakerbeam_multi_enroll.py  (multi-enrollment pooling)
│   └── ...                        (more training/eval scripts)
├── src/cafe_tse/                  (CAFE-TSE prototype + shared utilities)
├── gui/                           (Interactive Web demo)
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── assets/
├── demo_audio/                    (3 demo cases: mixture, target, baseline, ours)
├── results/
│   ├── data_sources.md            (data provenance and licenses)
│   └── summary.md                 (experiment summary)
└── requirements.txt
```

## Model Checkpoint

The experiments use the **mid fine-tune distill** model:
`open_speakerbeam_shared_clean80_student_mid_distill_ft_continue_w005/best.pt`

Test set (shared-clean80, 800 samples): SI-SDR 10.59 dB, SI-SDRi 10.59 dB, SDR 10.60 dB.

## Scope

The system models selective attention and target speaker voiceprint binding via enrollment conditioning. It does NOT explicitly model spatial source localization, DOA, binaural cues, microphone arrays, or spatial angle labels.

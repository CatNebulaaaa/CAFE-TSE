# Effective Model Improvement Plan

## Current Decision

The project should pivot from the self-written `TFGridNetLite` separator to the open-source TD-SpeakerBeam implementation as the reliable separation backbone.

Evidence:

| System | Valid SI-SDR | Test SI-SDR | Test SI-SDRi | Test SDR | Test SIR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Self-written CAFE-TSE teacher | 1.59 | 1.06 | 1.27 | 1.06 | 4.56 |
| Open SpeakerBeam, balanced10 | 7.49 | 7.00 | 7.21 | 6.99 | 15.89 |
| Open SpeakerBeam, balanced40 | 5.52 | 5.45 | 5.65 | 5.44 | 11.07 |
| Open SpeakerBeam, shared clean80 | 10.38 | 10.32 | 10.32 | 10.34 | 22.23 |

The original data, SI-SDR loss, and evaluation pipeline are valid. The failed component is the self-written target-speaker conditioning and separator structure.

## New Main Goal

Use TD-SpeakerBeam as the effective target-speaker extraction backbone, then rebuild the original CAFE-TSE improvements around it:

1. Keep a verified baseline above 5 dB SDR/SI-SDR.
2. Push the main final model toward 10 dB+ SDR/SI-SDR.
3. Add complexity-aware efficient inference.
4. Add explainable enrollment-guided front-end processing.
5. Distill the strong model into a lightweight student only after the teacher is strong.
6. Keep all claims tied to leakage-free balanced data and wrong-enrollment sanity checks.

## Stage 0: Reliable Baseline

Status: completed on the remote server.

Backbone:

- Source: `third_party/speakerbeam`
- Upstream repository: `https://github.com/BUTSpeechFIT/speakerbeam`
- Model class: `models.td_speakerbeam.TimeDomainSpeakerBeam`
- Bridge script: `scripts/train_open_speakerbeam.py`

Data:

- `librispeech_tse_balanced10`: 10 speakers, 800 train, 150 valid, 150 test.
- `librispeech_tse_balanced40`: 40 speakers, 2400 train, 400 valid, 400 test.
- Enrollment is disjoint from target and selected from the same split pool.
- Manifest now stores source paths for target, enrollment, and interferer.

Required sanity checks:

| Check | Expected |
| --- | --- |
| correct enrollment | high SI-SDR |
| wrong enrollment | clear degradation |
| interferer as enrollment | very poor output |
| target as enrollment | close to correct enrollment |
| zero enrollment | poor output |

Observed on SpeakerBeam balanced10:

| Enrollment | SI-SDR |
| --- | ---: |
| correct | 7.36 |
| wrong | -5.73 |
| target as enrollment | 7.47 |
| interferer as enrollment | -15.29 |
| zero | -3.71 |

This proves the effective model is doing target-speaker extraction rather than generic enhancement.

## Stage 0.5: 10 dB+ Quality Target

The current 7 dB result is a good debugging baseline, but it should not be treated as the final quality ceiling. A stronger target is:

| Level | Target | Meaning |
| --- | ---: | --- |
| Course pass | 5 dB+ | Meets the reference-guide threshold. |
| Strong final result | 10 dB+ | More convincing for a clean two-speaker TSE task. |
| Stretch | 12 dB+ | Approaches a mature clean-speech separation setting. |

Likely reasons the current open SpeakerBeam result stops around 5-7 dB:

1. The current `mid` model is smaller than the upstream default SpeakerBeam/Conv-TasNet scale.
2. The balanced40 data uses only 2400 training mixtures from `dev-clean`; it is enough to prove correctness but not ideal for a high-score teacher.
3. The synthesized mixtures include additive noise by default, so clean and noisy test results should be reported separately.
4. Training stopped early during the balanced40 run at epoch 12; it was not a converged quality run.

Quality-upgrade experiments:

| Experiment | Purpose | Expected outcome |
| --- | --- | --- |
| `open_speakerbeam_balanced10_strong` | Estimate model capacity ceiling on easier 10-speaker data | Should approach or exceed 10 dB if architecture/data are sufficient. |
| `open_speakerbeam_balanced40_strong` | Main 40-speaker teacher run | Target 8-10 dB first, then tune. |
| `open_speakerbeam_balanced40_clean` | Remove additive noise to isolate separation quality | Should be higher than noisy balanced40. |
| `open_speakerbeam_full_data` | Use more LibriSpeech/LibriMix data if available | Best path to stable 10 dB+. |

Observed follow-up:

| Experiment | Result | Interpretation |
| --- | --- | --- |
| `open_speakerbeam_balanced10_strong`, epoch 32 | Test SI-SDR 7.12 | Larger default-scale model did not beat the mid model quickly. |
| `open_speakerbeam_balanced10_mid` resumed to epoch 100, lr=3e-4 | Train SI-SDR 12.52, Test SI-SDR 7.30 | Continued training overfits the 800-sample balanced10 set; 10 dB+ needs more/better data or a clean/noisy split, not only more epochs. |
| `open_speakerbeam_shared_clean80_mid`, epoch 80 | Valid SI-SDR 10.38, Test SI-SDR 10.32, Test SDR 10.34 | 10 dB+ is reachable with clean mixtures, 80 shared speakers, 8000 train mixtures, and low-lr fine-tuning. |

Initial strong configuration:

```bash
--n_filters 512
--bn_chan 128
--hid_chan 512
--skip_chan 128
--adapt_enroll_dim 128
--n_blocks 8
--n_repeats 3
--i_adapt_layer 7
--batch_size 2
--epochs 80
--lr 0.001
```

The strong run should save every epoch and report both best validation and final test metrics. Current evidence shows that training-set SI-SDR can exceed 10 dB, but validation/test stay around 7 dB on balanced10. With a clean shared-speaker 80-speaker dataset, validation/test both exceed 10 dB. The main quality bottleneck is therefore data scale/split design and mixture difficulty, not the loss or model implementation.

## Stage 1: Rebuild Original Innovation Claims On SpeakerBeam

### 1. Enrollment-Guided Spectral Preprocessing

Original idea: use enrollment spectral statistics to guide the mixture front-end.

New implementation target:

- Apply EGSP as a deterministic preprocessing layer before SpeakerBeam.
- Keep it non-parametric first: profile from enrollment magnitude spectrum, normalized and clipped.
- Evaluate strength sweep on balanced40.
- Keep a wrong-enrollment EGSP sanity test. If wrong enrollment improves similarly to correct enrollment, report EGSP as generic enhancement rather than target-aware extraction.

Success criterion:

- SI-SDR does not drop below the SpeakerBeam baseline.
- Correct-enrollment EGSP improves hard samples or noisy samples.
- Wrong-enrollment EGSP does not become the main result if it breaks target-speaker specificity.

### 2. Complexity-Aware Inference

Original idea: dynamic routing based on mixture complexity.

New implementation target:

- Do not early-exit the open model blindly.
- First profile samples by complexity score, SI-SDR, RTF, and SIR.
- Train or evaluate smaller SpeakerBeam variants:
  - full: `n_filters=256, bn=64, hidden=256, repeats=2, blocks=6`
  - medium: lower hidden or repeats.
  - lite: lower filters and channels.
- Use complexity to select full/medium/lite models at inference.

Success criterion:

- Ours-Fast reduces average RTF/MAC proxy while keeping SI-SDR within about 0.5 dB of full SpeakerBeam.
- The route distribution is explainable by complexity bins.

### 3. Robustness Suite

Original idea: test restaurant/party-like noise and enrollment robustness.

New implementation target:

- Keep clean balanced40 as the main score.
- Add noise variants:
  - babble-like noise at 10 dB and 5 dB SNR.
  - shorter enrollment: 1 s and 2 s.
  - shuffled enrollment.
- Report quality and failure modes.

Success criterion:

- Correct enrollment remains clearly better than shuffled enrollment.
- The model degrades gracefully under noise.

## Stage 2: Distillation After A Strong Teacher

The previous distillation used a weak teacher, so the student inherited weak behavior.

New implementation target:

- Teacher: open SpeakerBeam clean/shared-speaker best checkpoint, then a larger strong-teacher checkpoint if it improves over the current 10.32 dB SI-SDR result.
- Student options:
  - smaller SpeakerBeam from the same open implementation.
  - self-written CAFE-TSE only as a student after it is supervised by a strong teacher.
- Distillation losses:
  - supervised SI-SDR to target.
  - teacher SI-SDR or waveform L1 to teacher output.
  - optional spectral loss.

Success criterion:

- Student gets close to teacher quality while reducing parameter count or runtime.
- Distillation baseline must compare against the same student trained without teacher.

Observed result on the first strong-teacher distillation attempt:

| System | Test SI-SDR | Test SDR | Test SIR | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Strong teacher, shared clean80 | 12.52 | 12.54 | 27.05 | New best teacher; strong model route works. |
| Small student, supervised | 8.16 | 8.16 | 17.30 | Same small architecture, trained directly to clean target. |
| Small student, from-scratch distillation | 8.02 | 8.02 | 16.76 | Did not beat supervised student; treat as a negative result. |

Failure interpretation:

- The student and distilled student used the same architecture, so the drop is not caused by a smaller distilled model.
- The task has clean waveform targets, so direct supervised SI-SDR to the target is already a strong training signal.
- Teacher waveform output is not ground truth; it contains residual errors. A high teacher weight can pull the student toward teacher artifacts.
- The first distillation run started from scratch with `teacher_weight=0.5`, which is too aggressive for this waveform regression setting.
- The first implementation also selected `best.pt` by mixed distillation loss instead of target validation SI-SDR. This has been fixed.

Improved distillation route:

1. Initialize the student from the supervised student checkpoint.
2. Use the strong teacher only as a low-weight regularizer.
3. Lower the learning rate for fine-tuning.
4. Select best checkpoint by validation SI-SDR against the clean target, not by mixed teacher loss.

Recommended command-level settings:

```bash
--init_student_checkpoint experiments/open_speakerbeam_shared_clean80_student_small/best.pt
--teacher_checkpoint experiments/open_speakerbeam_shared_clean80_strong/best.pt
--teacher_weight 0.1
--target_weight 1.0
--lr 0.0003
```

### Current Execution Route

The next work should be staged so that each change has a clean baseline and a clear failure mode.

| Step | Run | Goal | Keep only if |
| --- | --- | --- | --- |
| 1 | `shared-clean80-strong` | Train a larger SpeakerBeam teacher on the clean 80-speaker shared dataset to push beyond the current 10.32 dB test SI-SDR. | Test SI-SDR/SDR improves or gives a useful capacity ceiling. |
| 2 | `shared-clean80-student-small` | Train the exact small student without distillation. | It provides a fair efficiency baseline. |
| 3 | `shared-clean80-distill-small` | Distill the small student from the current 10 dB+ teacher. | Student beats the supervised small baseline at similar size/runtime. |
| 4 | EGSP / frequency-domain preprocessing | Add enrollment-guided spectral preprocessing behind explicit CLI flags. | Correct enrollment improves hard/noisy samples and wrong enrollment does not improve similarly. |
| 5 | enrollment robustness | Add short, noisy, shuffled, wrong-speaker, and interferer-enrollment evaluation. | Correct enrollment stays clearly separated from wrong/interferer enrollment. |
| 6 | complexity-aware routing | Route full/medium/lite models by complexity score. | Runtime/parameter proxy drops while SI-SDR stays close to the strong model. |

This order intentionally protects the current success: the 10 dB+ SpeakerBeam checkpoint remains the anchor, and every later improvement is allowed to fail independently without corrupting the working baseline.

## Stage 3: Report Narrative

Recommended final naming:

| Name | Meaning |
| --- | --- |
| Baseline | open TD-SpeakerBeam |
| Ours | SpeakerBeam + validated front-end or robustness module |
| Ours-Fast | complexity-aware model selection or distilled student |
| Failed Prototype | original TFGridNetLite CAFE-TSE |

The report should honestly explain that the original hand-written model became a failure case and debugging evidence, while the final working system uses an open verified TSE backbone plus system-level improvements.

## Immediate Next Steps

1. Start `shared-clean80-strong` as the strong-teacher capacity run.
2. If it beats the current 10.32 dB test SI-SDR checkpoint, use it as the teacher; otherwise keep `open_speakerbeam_shared_clean80_mid/best.pt`.
3. Run `shared-clean80-student-small` and `shared-clean80-distill-small` for a fair student comparison.
4. Implement EGSP preprocessing for `scripts/train_open_speakerbeam.py` behind CLI flags.
5. Run enrollment and frequency-domain ablations only after the teacher/student baselines are stable.

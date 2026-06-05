# Reference Guide Follow-up

This note tracks the changes prompted by `作业要求/Speech Separation Project（参考指南）.pdf`.

## Guide Requirements Mapped to Current Work

| Guide item | Current status |
| --- | --- |
| Real mixed-speech data | Done. MiniLibriMix real-data subset: 800 train, 100 validation, 100 test mixtures. |
| At least 3 speakers and 50 test samples | Done. Test split contains 100 mixtures and 98 target speakers. |
| Target speaker extraction with reference speech | Done. The system uses mixture + enrollment and evaluates 1s/3s/5s enrollment settings. |
| STFT, mask/separation core, reconstruction | Done. CAFE-TSE uses STFT, target-conditioned TF-GridNet-Lite, and iSTFT reconstruction. |
| SDR/PESQ style evaluation | Partly done. SDR/SIR/SAR/SI-SDRi and PESQ are computed. STOI is unstable for direct model-output evaluation and should not be used as a main reported metric unless fully fixed. |
| Demo audio | Done. `demo_audio/` contains 3 groups: mixture, target, baseline, and Ours. |
| Failure-case discussion | Needs report update. Use noisy enrollment degradation and babble-mixture absolute SDR degradation as two explicit failure cases. |
| Real/noisy cocktail-party robustness | Added. Babble-noise mixture variants at 5 dB and 10 dB SNR were evaluated. |

## New Experiments Completed

| Experiment | Purpose | Key outcome |
| --- | --- | --- |
| `mini_exp25_baseline_egsp_quality` | Probe whether 6-block baseline + EGSP beats Ours | It only slightly improves SI-SDRi/PESQ but restores baseline-level params/MACs, so it is not a better final method. |
| `guide_ours_enroll_*` | Verify 1s/3s/5s and noisy enrollment robustness | Clean enrollment improves with longer duration; noisy enrollment strongly degrades performance. |
| `guide_*_mixture_babble_*` | Evaluate cocktail-party-like background speech noise | Ours keeps higher SI-SDRi than baseline under babble noise, but absolute SDR remains negative in noisy mixtures. |
| `mini_exp28_ours_egsp_target_finetune_full` | Train the 5-block Ours model with EGSP active during fine-tuning | Completed on the remote server. It improves Ours from 0.4980 to 0.5088 SI-SDRi while keeping the 5-block parameter/MAC budget. |
| `mini_exp29_base_egsp_target_finetune_full` | Train the 6-block EGSP quality upper-bound model | Completed on the remote server. It reaches 0.5154 SI-SDRi, but requires baseline-level params/MACs. |

## Current Breakpoint

The old local note said `mini_exp26_ours_egsp_target_finetune` was running and `mini_exp27_base_egsp_target_finetune` was pending. The remote server shows that this work was superseded by the fuller `run_minilibrimix_quality_boost.sh` script, which completed `mini_exp28_ours_egsp_target_finetune_full` and `mini_exp29_base_egsp_target_finetune_full`.

The experiment breakpoint is therefore no longer a training job. The next work item is report integration: update the final tables and interpretation with `results/summary_mini_quality_boost.md`, while keeping `mini_exp18_egsp_spec_s005_selected` as the simple inference-time EGSP method and using `mini_exp28_ours_egsp_target_finetune_full` as the best 5-block quality method.

## Current Interpretation

The strongest defensible final method is now the 5-block EGSP target fine-tuned model, `mini_exp28_ours_egsp_target_finetune_full`, if the report wants the best quality at the lightweight budget. The simpler inference-only EGSP model, `mini_exp18_egsp_spec_s005_selected`, remains useful as an ablation because it shows that most of the gain comes from EGSP without retraining.

The final leakage audit found that the earlier reference-anchor result is not valid as a main result under disjoint enrollment. After fixing target/enrollment leakage, s2 speaker parsing, cross-frequency modeling, and condition extraction, the best disjoint retrain is `mini_exp43_student_magmask_disjoint_tfgridfix_lr1e3` with SI-SDRi=0.0710 and SDR=-0.0352. The report should therefore frame CAFE-TSE as a complete lightweight target-extraction prototype and debugging study, not as a high-fidelity model that reaches the guide's ideal 5 dB target.

# Project Notes

This folder contains the current project decision records.

## Current Core Documents

| Document | Purpose |
| --- | --- |
| `MODEL_ROOT_CAUSE_FINDINGS.md` | Debugging evidence showing why the self-written CAFE-TSE teacher failed and why open SpeakerBeam is the effective backbone. |
| `EFFECTIVE_MODEL_IMPROVEMENT_PLAN.md` | Step-by-step plan for improving the effective SpeakerBeam-based system, including the new 10 dB+ target. |

## Current Direction

1. Use open TD-SpeakerBeam as the reliable target-speaker extraction backbone.
2. Treat the self-written `TFGridNetLite` system as a failed prototype and diagnostic baseline.
3. Improve quality first, targeting 10 dB+ SDR/SI-SDR.
4. Add CAFE-TSE-style system improvements after the strong teacher is established:
   - enrollment-guided preprocessing,
   - complexity-aware inference,
   - robustness evaluation,
   - distillation into a smaller student.

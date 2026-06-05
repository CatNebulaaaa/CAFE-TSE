# Submission Code Requirements Check

## Direct Requirements Found

The official course files do not give a separate packaging template for code submission, but they do contain explicit code-related grading requirements.

Sources checked:

- `作业要求/课程报告题目-认知与计算-2026.pdf`
- `作业要求/课程报告评分标准-认知与计算-2026.pdf`
- `作业要求/课程报告的编写格式规范-认知与计算(1).pdf`
- `作业要求/课程报告模板-认知与计算(1).docx`
- Existing project audit notes based on `Speech Separation Project（参考指南）.pdf`

## Explicit Code-Related Wording

From `课程报告评分标准-认知与计算-2026.pdf`, Task 2:

| Rubric item | Code implication |
| --- | --- |
| “模型设计与实现（30%）：架构清晰合理……代码/算法实现完整，具有创新性。” | Code or at least complete algorithm implementation must be provided or clearly documented. |
| “逻辑结构与表达（10%）：图表清晰规范，代码注释完整，格式符合学术报告要求。” | Submitted code should have readable comments and clear organization. |

This means code is not merely optional for Task 2. It is part of the grading dimensions.

## Requirements Inferred From The Speech Separation Guide

The PDF guide is image-based, but the project audit and earlier extraction notes identify the following expected deliverables:

| Expected material | Current project status |
| --- | --- |
| Complete source code | `src/`, `scripts/`, `run_*.sh` exist and are committed. |
| README / usage instructions | `README.md`, `README_CAFE-TSE.md`, `README_CAFE-TSE_CodexHarness.md` exist. |
| Training and evaluation scripts | `scripts/train_open_speakerbeam.py`, `scripts/train_open_speakerbeam_distill.py`, `run_open_speakerbeam_quality.sh`, `run_open_speakerbeam_teacher_student_pipeline.sh` exist. |
| Model implementation | Original CAFE-TSE modules under `src/cafe_tse/`; effective SpeakerBeam bridge under `scripts/`. |
| Experiment results / figures | Current report and docs contain result tables; final results still pending for strong teacher/student/distillation. |
| Demo audio examples | Existing audit says `demo_audio/` contains 3 groups, but this folder should be verified before final packaging. |
| Failure-case discussion | Added in report draft and docs. |
| Code comments | Basic code organization exists; before final submission, add a short README section explaining which scripts reproduce main results. |

## Recommended Final Submission Package

Use a clean package such as:

```text
认知与计算课程报告_王志翔/
  report/
    final_integrated_report.pdf
    final_integrated_report.tex
  src/
    cafe_tse/
  scripts/
    train_open_speakerbeam.py
    train_open_speakerbeam_distill.py
    evaluate_enrollment_anchor.py
    audit_tse_manifest.py
  configs/
  run_open_speakerbeam_quality.sh
  run_open_speakerbeam_teacher_student_pipeline.sh
  README.md
  requirements.txt
  docs/
    MODEL_ROOT_CAUSE_FINDINGS.md
    EFFECTIVE_MODEL_IMPROVEMENT_PLAN.md
    SUBMISSION_CODE_REQUIREMENTS.md
  demo_audio/
    case_*/mixture.wav
    case_*/target.wav
    case_*/estimate.wav
```

Do not include:

- Raw LibriSpeech/LibriMix datasets if they are too large.
- Full training checkpoints unless the teacher explicitly asks for model weights.
- Temporary logs, `.aux`, `.out`, `.log`, `__pycache__`, or large intermediate experiment folders.

## README Must Include

Before final packaging, make sure README has:

1. Environment setup.
2. How to prepare or locate data manifests.
3. How to run the effective SpeakerBeam baseline.
4. How to run strong teacher training.
5. How to run student supervised training and distillation.
6. How to evaluate metrics.
7. Where final report figures and result tables come from.
8. A note that raw datasets/checkpoints are excluded if not submitted.

## Bottom Line

Yes, there is a code requirement. The scoring standard explicitly mentions complete code/algorithm implementation and complete code comments for Task 2. The final submission should include source code, scripts, README, requirements, report PDF/source, and demo materials, while excluding raw datasets and large checkpoints unless requested.

# Final Integrated Course Report Structure Plan

## Report Positioning

Final report title:

> 《认知与计算》课程报告：从人类智能对比到鸡尾酒会语音分离系统的认知计算建模

The final submission should be one integrated paper, not three unrelated answers. The recommended narrative is:

1. Task 1 explains the broad difference between human intelligence and artificial intelligence.
2. Task 2 uses the cocktail-party problem as a concrete cognitive-computing case study.
3. Task 3 returns from the system case study to the psychological basis of machine learning, using neural networks and backpropagation as the main example.

This keeps the three required questions independent enough for grading, while giving the whole report a single academic thread: cognition -> computation -> implemented model -> learning mechanism.

## Required Format

Follow `作业要求/课程报告的编写格式规范-认知与计算(1).pdf` and `作业要求/课程报告模板-认知与计算(1).docx`.

- Cover page: use the template fields: name, student ID, grade, major direction, instructor, year/month.
- Main title: centered, size-3 Heiti.
- Student information below title: name in size-4 Songti; grade/major/student ID in small size-5 Songti with parentheses.
- Body text: size-5 Songti.
- Section numbering: `1`, `1.1`, `1.1.1`, `(a)`, `(1)`.
- Main section headings: size-4 Songti.
- Subsection headings: size-5 Heiti.
- References: numbered list at the end, small size-5 Songti.
- Figures and tables: number consecutively as `图1`, `图2`, `表1`, `表2`; captions placed below figures/tables.
- Page numbers: centered.

## Suggested Length Allocation

| Part | Score Weight | Suggested Pages | Reason |
| --- | ---: | ---: | --- |
| Front matter | - | 1-2 | Cover, title page, abstract, keywords. |
| Task 1: AI vs human intelligence | 15 | 3-4 | Needs breadth, theory, and critical comparison, but should not dominate. |
| Task 2: Cocktail-party speech separation | 60 | 9-12 | Main project, strongest scoring weight, needs method, experiment, figures, results, discussion. |
| Task 3: ML psychology and human learning | 25 | 4-5 | Needs algorithm detail, cognitive mapping, and limitations. |
| Conclusion and references | - | 1-2 | Integrated conclusion and unified bibliography. |

Recommended final length: about 18-23 pages, excluding cover if needed. If page limit becomes tight, compress Task 1 and Task 3 first, not Task 2.

## Full Paper Structure

### Cover

Use the official Word template layout.

Fields:

- Course report
- Name
- Student ID
- Grade
- Major direction
- Instructor
- Year/month

### Title Page

Title:

> 《认知与计算》课程报告：从人类智能对比到鸡尾酒会语音分离系统的认知计算建模

Below title:

- Student name
- Grade, major, student ID

### Abstract

One integrated abstract, not three separate abstracts.

Recommended content:

1. State that the report discusses three cognitive-computing questions.
2. Summarize Task 1: AI exceeds humans in closed, data-rich tasks but remains weaker in embodied cognition, causality, emotion, value judgment, and lifelong learning.
3. Summarize Task 2: the report implements a target-speaker extraction system for the cocktail-party problem using mixture speech plus enrollment speech.
4. Use current verified result: open TD-SpeakerBeam on clean shared 80-speaker data reaches about 10.32 dB test SI-SDR and 10.34 dB SDR; the earlier self-written model is analyzed as a failure case.
5. Summarize Task 3: neural networks/backpropagation partly map to connectionism and feedback learning, but differ from biological and psychological learning in data efficiency, agency, and continual adaptation.

Keywords:

`认知与计算；人工智能；人类智能；鸡尾酒会问题；目标说话人提取；神经网络；反向传播`

## 1 Artificial Intelligence And Human Intelligence

Source to reuse: `report/report_tasks_1_3.tex`, section `人工智能与人类智能的多维度对比分析`.

This part answers Task 1 and should be written as a small essay.

### 1.1 Problem Introduction: Beyond A Simple Strong/Weak Comparison

Purpose:

- Explain why comparing AI and human intelligence cannot be reduced to one score.
- Introduce multidimensional comparison.

Must cover:

- AI can exceed humans in some benchmarked tasks.
- Human intelligence remains broader because it includes embodied experience, social cognition, values, and adaptation.

### 1.2 Cognitive Architecture: Symbolic, Connectionist, And Embodied Systems

Compare:

- Human intelligence: brain-body-environment system.
- AI: symbolic systems, neural networks, large-scale statistical models.

Scoring target:

- Covers cognitive architecture and information-processing mechanism.

### 1.3 Information Processing: Statistical Correlation And Meaning Understanding

Compare:

- AI learns statistical regularities from data.
- Humans integrate perception, action, context, and lived meaning.

Critical point:

- AI can use language functionally, but functional language use is not identical to subjective understanding.

### 1.4 Learning And Generalization

Compare:

- AI: large data, gradient optimization, benchmark generalization.
- Humans: small-sample learning, analogy, causal models, transfer from experience.

Can mention:

- Foundation models and multimodal models.
- Human infant learning and commonsense generalization.

### 1.5 Reasoning, Causality, And Common Sense

Discuss:

- AI progress in mathematics, code, and formal reasoning.
- Weakness in causal grounding, real-world intervention, and robust common sense.

### 1.6 Emotion, Social Intelligence, And Value Judgment

Discuss:

- AI can recognize and simulate emotional language.
- Human emotion is embodied, motivational, and socially situated.
- AI lacks intrinsic value judgment and responsibility.

### 1.7 Creativity And Efficiency

Discuss:

- AI creativity as recombination and fast generation.
- Human creativity as problem framing, value selection, aesthetic judgment.
- Energy efficiency and embodied adaptation.

### 1.8 Summary Of AI Strengths And Limits

Recommended table:

`表1 人工智能与人类智能的多维度对比`

Columns:

- Dimension
- AI advantage
- Human advantage
- Boundary or open problem

## 2 Cocktail-Party Problem And Target Speaker Extraction System

Source to reuse:

- Old structure: `report/CAFE_TSE_course_paper.tex`
- Current correct evidence: `docs/MODEL_ROOT_CAUSE_FINDINGS.md`
- Improvement route: `docs/EFFECTIVE_MODEL_IMPROVEMENT_PLAN.md`

This is the central part of the report and should be the longest section.

Important narrative update:

- Do not present the original self-written CAFE-TSE/TFGridNetLite as the final successful model.
- Present it as an early prototype and failure/debugging case.
- Present the verified open-source TD-SpeakerBeam backbone as the effective final baseline.
- Present system improvements as staged extensions: strong teacher, distillation, enrollment sanity checks, frequency-domain preprocessing, and complexity-aware routing.

### 2.1 Cognitive Background: The Cocktail-Party Problem

Purpose:

- Explain Cherry's cocktail-party problem.
- Connect it to selective attention, auditory scene analysis, feature binding, and target tracking.

References:

- Cherry 1953
- Broadbent 1958
- Bregman 1990

Recommended figure:

`图1 鸡尾酒会问题中的认知机制与计算模块映射`

Mapping:

- Selective attention -> enrollment/reference speech.
- Auditory object formation -> speech separation backbone.
- Feature binding -> speaker-conditioned extraction.
- Resource allocation -> lightweight student/distillation/complexity routing.

### 2.2 Task Definition

Define target speaker extraction:

- Input: mixture speech `x(t)` and enrollment/reference speech `e(t)`.
- Output: target estimate `\hat{s}(t)`.
- Objective: maximize SI-SDR/SDR and preserve target identity.

Clarify why this is closer to the course requirement than blind source separation:

- The model must know "which speaker to listen to."

### 2.3 Dataset Construction And Leakage Audit

Must include:

- Mixture generation.
- Target/enrollment/interferer roles.
- Disjoint target and enrollment utterances.
- Correct/wrong/interferer enrollment sanity checks.
- Clean shared 80-speaker data used for the current 10 dB+ result.

Recommended table:

`表2 数据集与划分设置`

Rows:

- balanced10
- balanced40
- shared clean80

Columns:

- Train/valid/test samples
- Number of speakers
- Noise condition
- Enrollment condition
- Purpose

Key result to mention:

- The dataset/loss/evaluation pipeline is valid.
- The earlier failure came mainly from model implementation, not from the evaluation metric.

### 2.4 Model Development History

This section should honestly explain the project evolution.

Recommended subsections:

#### 2.4.1 Early Self-Written CAFE-TSE Prototype

Describe:

- STFT/mask-based design.
- EGSP idea.
- Target speaker conditioning.
- TFGridNetLite / sparse fusion / dynamic inference.

But state clearly:

- Under strict disjoint enrollment, this model did not reach high-quality separation.
- It is kept as an engineering prototype and failure analysis.

#### 2.4.2 Root Cause Analysis

Use `docs/MODEL_ROOT_CAUSE_FINDINGS.md`.

Key points:

- Teacher model only reached around 1 dB test SI-SDR.
- Wrong/target enrollment probes showed that the self-written conditioning/separator was unreliable.
- Open SpeakerBeam immediately reached much higher performance, proving the data and loss were not the main issue.

Recommended table:

`表3 模型排错证据`

Rows:

- Self-written CAFE-TSE teacher
- Open SpeakerBeam balanced10
- Open SpeakerBeam balanced40
- Open SpeakerBeam shared clean80

Columns:

- Valid SI-SDR
- Test SI-SDR
- SDR
- SIR
- Interpretation

Current key values:

- Self-written teacher: about 1.06 dB test SI-SDR.
- Open SpeakerBeam balanced10: about 7.00 dB test SI-SDR.
- Open SpeakerBeam shared clean80: 10.32 dB test SI-SDR, 10.34 dB SDR, 22.23 dB SIR.

#### 2.4.3 Final Effective Backbone: Open TD-SpeakerBeam

Explain:

- Use a verified open-source target speaker extraction implementation.
- Architecture: time-domain speaker-conditioned separation.
- Enrollment controls which speaker is extracted.

Important phrasing:

- "The final reliable system uses TD-SpeakerBeam as the separation backbone, while the original CAFE-TSE components are reframed as system-level improvement directions."

### 2.5 Current System Architecture

Recommended figure:

`图2 最终目标说话人提取系统结构`

Pipeline:

1. Mixture waveform.
2. Enrollment waveform.
3. Speaker encoder / adaptation pathway.
4. TD-SpeakerBeam separator.
5. Target speech estimate.
6. Evaluation module.

Optional extension blocks shown with dashed lines:

- EGSP frequency-domain preprocessing.
- Strong teacher.
- Student distillation.
- Complexity-aware routing.

### 2.6 Training Strategy

Describe:

- Supervised SI-SDR loss.
- Strong teacher training on clean shared 80-speaker data.
- Student baseline training.
- Distillation from teacher to student.

Recommended figure:

`图3 强教师-学生蒸馏训练流程`

Current execution route:

1. Train strong teacher.
2. Compare strong teacher with current 10 dB+ mid teacher.
3. Select the better teacher.
4. Train small student without distillation.
5. Train small student with teacher supervision.
6. Compare quality/efficiency.

### 2.7 Experimental Setup

Must satisfy scoring standard:

- Different data sizes.
- Clean/noisy conditions where available.
- Correct/wrong/interferer enrollment sanity checks.
- Metrics: SI-SDR, SI-SDRi, SDR, SIR, SAR, PESQ/STOI if stable.
- Efficiency: parameter count, runtime or MAC proxy if available.

Recommended table:

`表4 实验设置与评估指标`

Metrics:

- SI-SDR: main separation quality.
- SDR/SIR/SAR: BSS-style decomposition.
- PESQ/STOI: optional perceptual references.
- Wrong-enrollment degradation: target-specificity sanity check.
- Runtime/parameter proxy: efficiency.

### 2.8 Main Results

Use current successful result as main anchor.

Recommended table:

`表5 主要实验结果`

Rows:

- Self-written CAFE-TSE teacher
- Open SpeakerBeam balanced10
- Open SpeakerBeam balanced40
- Open SpeakerBeam shared clean80
- Strong teacher, if completed
- Student supervised, if completed
- Student distilled, if completed

Columns:

- Valid SI-SDR
- Test SI-SDR
- SI-SDRi
- SDR
- SIR
- Params/runtime if available

Current conclusion:

- The project now has a credible 10 dB+ target-speaker extraction result.
- This is a major improvement over the earlier failed self-written model.

### 2.9 Enrollment And Robustness Analysis

Use sanity checks:

- Correct enrollment.
- Wrong enrollment.
- Interferer enrollment.
- Target-as-enrollment.
- Zero enrollment.

Recommended table:

`表6 Enrollment sanity check`

Known balanced10 values:

- Correct: about 7.36 dB SI-SDR.
- Wrong: about -5.73 dB SI-SDR.
- Interferer: about -15.29 dB SI-SDR.
- Target-as-enrollment: about 7.47 dB SI-SDR.
- Zero: about -3.71 dB SI-SDR.

Interpretation:

- The model is performing target speaker extraction, not generic denoising.

### 2.10 Ablation And Improvement Plan

This is where previous failed improvements should be handled carefully.

Subsections:

- EGSP / frequency-domain preprocessing.
- Complexity-aware routing.
- Knowledge distillation.
- Robustness to noisy/short/shuffled enrollment.

Important rule:

- Do not overclaim incomplete improvements.
- Say these are staged extensions around the verified SpeakerBeam backbone.

### 2.11 Failure Cases And Lessons

Must include because it strengthens the report.

Cases:

1. Self-written separator failed under strict enrollment.
2. Too-small dataset caused overfitting and limited SDR.
3. Wrong or leaked enrollment can create misleading high scores.
4. Stronger model may not always improve without enough data/regularization.

Recommended table:

`表7 失败案例、原因与修复`

This section helps satisfy critical thinking, theoretical discussion, and experiment-analysis criteria.

### 2.12 Cognitive Interpretation

Connect the system back to cognition:

- Enrollment = selective attention cue.
- Speaker conditioning = target identity binding.
- Separation backbone = auditory scene analysis.
- Distillation/complexity routing = limited cognitive resource allocation.

Also discuss differences from humans:

- Human auditory attention is online, multimodal, adaptive, and context-aware.
- The model depends on supervised data and fixed training distribution.

## 3 Psychological Basis Of Machine Learning Algorithms

Source to reuse: `report/report_tasks_1_3.tex`, section `机器学习算法的心理学基础与人类学习过程的映射`.

This part answers Task 3 as a second small essay.

Recommended algorithm:

- Neural networks and backpropagation.

Reason:

- Neural networks are central to modern AI and directly connected to connectionism.
- Backpropagation connects naturally to feedback learning, error correction, representation learning, and debates about biological plausibility.
- The cocktail-party model in Task 2 also uses neural networks, so the whole paper stays coherent.

### 3.1 Algorithm Choice And Problem Positioning

Explain why neural networks/backpropagation are selected.

### 3.2 Neural Networks And Connectionist Cognitive Theory

Map:

- Neurons/units.
- Connections/weights.
- Distributed representations.
- Hierarchical feature learning.

### 3.3 Backpropagation And Feedback Learning

Explain:

- Forward prediction.
- Loss/error.
- Gradient-based weight update.

Psychological mapping:

- Feedback correction.
- Reinforcement from mistakes.
- Practice-driven improvement.

### 3.4 Relation To Hebbian Learning

Compare:

- Hebbian learning: local association, "cells that fire together wire together."
- Backpropagation: global objective and credit assignment.

Critical point:

- Backpropagation is useful computationally but not fully biologically realistic.

### 3.5 How Neural Networks Simulate Human Learning

Subpoints:

- From features to concepts.
- Feedback-based correction.
- Transfer learning.
- Curriculum learning.

Connect to Task 2:

- The speech-separation model learns from many mixture-target pairs.
- Distillation resembles learning from a stronger teacher.

### 3.6 How Neural Networks Simulate Decision Processes

Discuss:

- Internal representation.
- Output choice.
- Confidence/probability.
- Task-specific optimization.

### 3.7 What Neural Networks Still Fail To Simulate

Must cover scoring requirements:

- Small-sample learning.
- Active learning and curiosity.
- Embodied semantic understanding.
- Continual learning and catastrophic forgetting.
- Social and value-based decision-making.

### 3.8 Critical Summary

Conclusion:

- Neural networks are powerful functional models of learning, but only partial models of human cognition.
- Similarity is useful as an analogy, not as identity.

Recommended table:

`表8 神经网络学习与人类学习的对应关系及差异`

## 4 Integrated Discussion

This section makes the whole report feel like one paper.

### 4.1 From Cognitive Theory To Implemented Computation

Explain:

- Task 1 gives the broad contrast.
- Task 2 shows a concrete implemented cognitive-computing system.
- Task 3 explains the learning mechanism behind such systems.

### 4.2 What The Cocktail-Party Project Shows About AI And Human Intelligence

Key points:

- AI can perform a precise speech extraction task well when given data, architecture, and target cues.
- It remains brittle to data distribution, enrollment quality, and implementation details.
- This matches the Task 1 claim: AI is strong in optimized tasks but less flexible than humans.

### 4.3 Future Work

Use staged plan:

1. Finish strong teacher training.
2. Distill small student.
3. Add validated frequency-domain EGSP.
4. Add enrollment robustness and noisy scenes.
5. Add complexity-aware routing.
6. Extend to more realistic multi-speaker/noisy cocktail scenes.

## 5 Conclusion

One integrated conclusion.

Should include:

- AI vs human intelligence is multidimensional.
- The cocktail-party system demonstrates how selective attention can be computationally modeled.
- Verified SpeakerBeam-based target extraction now reaches 10 dB+ SI-SDR/SDR on clean shared 80-speaker data.
- The debugging process shows why rigorous data audit and sanity checks are essential.
- Neural networks/backpropagation provide a useful but incomplete computational analogy to human learning.

## References

Use one unified bibliography, not separate reference lists.

Recommended reference groups:

### AI And Human Intelligence

- Turing, M. A. Computing Machinery and Intelligence.
- Newell and Simon. Human Problem Solving.
- OpenAI. GPT-4 Technical Report.
- Relevant recent AI benchmark/capability papers already used in `report_tasks_1_3.tex`.

### Cognitive Science And Learning

- Rumelhart, Hinton, Williams. Backpropagation.
- Hebb. The Organization of Behavior.
- Bengio et al. Curriculum Learning.
- Cognitive learning / continual learning references from the existing Task 3 draft.

### Cocktail-Party And Speech Separation

- Cherry 1953.
- Broadbent 1958.
- Bregman 1990.
- Griffin and Lim 1984.
- Conv-TasNet.
- DPRNN.
- SepFormer.
- TF-GridNet.
- LibriMix.
- VoiceFilter.
- SpeakerBeam.
- SpEx+.
- Hinton et al. Knowledge distillation.
- Vincent et al. BSS metrics.

## Figure And Table Checklist

Recommended figures:

1. Cognitive mechanism to computational module mapping.
2. Final target speaker extraction architecture.
3. Strong-teacher/student distillation pipeline.
4. Main result bar chart.
5. Enrollment sanity check chart.
6. Spectrogram example: mixture, target, estimate.
7. Failure-case/debugging flow if space allows.

Recommended tables:

1. AI vs human intelligence comparison.
2. Dataset split and mixture settings.
3. Model debugging evidence.
4. Experimental metrics and setup.
5. Main results.
6. Enrollment sanity checks.
7. Failure cases and fixes.
8. Neural network vs human learning mapping.

## Writing Priority

If time is limited, write in this order:

1. Task 2 main system section, because it is 60 points and needs the newest experiment results.
2. Abstract and integrated conclusion.
3. Task 1 compression from existing `report_tasks_1_3.tex`.
4. Task 3 compression from existing `report_tasks_1_3.tex`.
5. Figures/tables.
6. References and format polish.

## Important Revision Notes For Existing Drafts

### `report/report_tasks_1_3.tex`

Keep the overall content, but compress it slightly so it does not crowd out Task 2.

Update any very recent factual claims only if citations are reliable. If uncertain, prefer stable examples such as image classification, speech recognition, AlphaFold, machine translation, and standardized benchmark performance.

### `report/CAFE_TSE_course_paper.tex`

Needs major narrative update:

- Replace the old final result based on the failed self-written model.
- Use open TD-SpeakerBeam as the verified effective model.
- Keep CAFE-TSE/EGSP/distillation as prototype and staged improvement directions.
- Add the 10.32 dB SI-SDR / 10.34 dB SDR clean shared80 result.
- Add strong-teacher/student-distillation training plan and latest results once finished.
- Preserve failure analysis because it is academically useful and honest.

## Final Thesis Sentence

Suggested core claim:

> 本文认为，认知计算系统的价值不在于简单复制人类智能，而在于将人类认知中的选择性注意、特征绑定、反馈学习和资源分配等机制转化为可实现、可验证、可反思的计算模块。人工智能在明确目标和充分数据下可以达到较高任务性能，但其泛化、具身理解和自主学习能力仍与人类智能存在本质差距。

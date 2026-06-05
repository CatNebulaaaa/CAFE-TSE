# CAFE-TSE 系统级创新方案

本文档用于梳理 CAFE-TSE 在“系统”层面的可创新模块，并给出后续实验优先级。当前主干创新已经形成稳定结果：`5-block distilled TF-GridNet-Lite` 在 MiniLibriMix 真实测试集上接近 baseline 指标，同时显著降低参数量、MACs 和 RTF。后续创新应围绕系统其它模块展开，避免只停留在分离主干。

## 1. 总体系统结构

CAFE-TSE 可以拆成 8 个模块：

1. 数据预处理与任务构建
2. 时频编码前端
3. 目标说话人条件建模
4. 条件融合策略
5. 轻量分离主干
6. 课程学习与蒸馏训练
7. 动态推理控制
8. 评估与可解释性分析

当前最值得采用的系统叙事是：

```text
Enrollment-guided spectral preprocessing
        +
USEF-style speaker conditioning
        +
Sparse / gated condition fusion
        +
5-block distilled TF-GridNet-Lite
        +
difficulty-aware evaluation and interpretability
```

## 2. 模块创新设计

### 2.1 数据预处理：Enrollment-Guided Spectral Preprocessing

目标：在进入分离模型前，用 enrollment 的长期频谱画像对 mixture 的频谱进行目标说话人相关增强。

动机：鸡尾酒会问题中的选择性听觉注意并不是无差别处理所有频率，而是会根据目标说话人的声学线索增强相关频带。该模块将目标说话人的长期频谱轮廓作为轻量先验，引导后续模型更容易关注目标声源。

设计：

```text
enrollment.wav -> STFT -> target spectral profile
mixture.wav    -> STFT -> target-aware frequency weighting
weighted spectrum / features -> CAFE-TSE separator
```

建议命名：

```text
EGSP: Enrollment-Guided Spectral Preprocessing
```

实验设置：

| 实验 | 目的 |
|---|---|
| 5-block distilled | 当前主结果 |
| 5-block distilled + EGSP | 检验频谱预处理是否提升或保持性能 |
| EGSP by difficulty | 检查 hard 样本是否更受益 |

应输出图表：

- enrollment 频谱画像图
- mixture 原始频谱 vs EGSP 加权频谱
- easy / medium / hard 分组指标表

优先级：高。该模块实现成本低，和系统主题强相关。

### 2.2 时频编码：Auditory-Inspired Multi-Resolution Encoding

目标：增强当前单一 STFT 表征，使模型同时关注短时语音变化和长时音色结构。

可选设计：

```text
Short STFT: n_fft=256  -> 捕捉瞬时辅音和重叠变化
Main STFT:  n_fft=512  -> 当前主分支
Long STFT:  n_fft=1024 -> 捕捉音色和谐波结构
```

更稳的实现方式是先加入 multi-resolution STFT loss，而不是立刻大改模型输入。

优先级：中。适合作为第二阶段增强，但当前不是最紧急。

### 2.3 条件模块：Speaker-Aware Gated Conditioning

目标：让模型学习在每个时频位置注入多少目标说话人条件，而不是固定强度注入。

设计：

```text
gate = sigmoid(W[mix_feature, speaker_condition])
fused_feature = mix_feature + gate * speaker_condition
```

预期优势：

- 减少 noisy / short enrollment 对模型的错误引导
- 在目标说话人显著区域增强条件
- 在干扰强或不确定区域保持自适应抑制

应输出图表：

- gate heatmap
- clean / noisy enrollment 鲁棒性表

优先级：高，但实现复杂度高于 EGSP。

### 2.4 条件融合：Dynamic Sparse Condition Fusion

当前系统已经实现 Sparse USEF Fusion，即只在部分 block 注入目标说话人条件。进一步创新可以让注入位置随样本复杂度变化。

设计：

```text
easy sample   -> inject at block [0]
medium sample -> inject at block [0, 2]
hard sample   -> inject at block [0, 2, 4]
```

优点：

- 比动态跳过主干 block 更稳
- 仍保留完整分离主干
- 有清楚的注意资源分配解释

优先级：中高。

### 2.5 分离主干：5-Block Distilled TF-GridNet-Lite

当前已经完成并作为主结果：

| 模型 | SI-SDRi | Params | MACs | RTF wall |
|---|---:|---:|---:|---:|
| baseline | 0.1591 | 288,818 | 10.94G | 0.002612 |
| 5-block distilled | 0.1560 | 168,562 | 6.34G | 0.001831 |

结论：5-block 蒸馏模型是当前最好的准确率-效率折中点。该模块不建议继续大改，后续应围绕系统其它部分做补充创新。

### 2.6 训练策略：Depth-Aware Curriculum Distillation

当前动态早退实验失败的主要原因是：模型训练时总是完整 5 block，推理时强行只跑 3/4 block，子路径没有被训练过。

解决方案：

```text
teacher: baseline 6-block
student: 5-block slimmable model

training:
  randomly choose active depth = 3 / 4 / 5
  each depth learns:
    1. true target speech
    2. teacher output
    3. full-depth student output
```

建议命名：

```text
Depth-Aware Curriculum Distillation
```

优先级：最高，但实验成本高于 EGSP。若时间允许，应作为动态推理失败后的主要改进方向。

### 2.7 推理控制：Confidence-Based Adaptive Inference

当前 complexity-based routing 只根据输入复杂度决策，无法判断中间输出是否已经足够好。未来可改为 confidence-based early exit。

可能信号：

- 输出与 enrollment 的 speaker similarity
- 相邻 block 输出变化
- 频谱残差收敛程度
- 非目标说话人残留检测

优先级：中。适合作为未来工作，不建议当前阶段强行实现。

### 2.8 评估与可解释性：Difficulty-Aware and Pareto Analysis

该模块必须完善，因为它决定系统作业的完整度和展示效果。

必须输出：

| 图/表 | 内容 |
|---|---|
| 主结果表 | baseline、dynamic、4-block、5-block distilled |
| 效率表 | Params、MACs、RTF、memory |
| Pareto 图 | x=MACs/RTF，y=SI-SDRi |
| 训练曲线 | epoch vs valid loss，说明 60 epoch 未收敛 |
| 模块消融表 | no curriculum / no distill / no sparse / 5-block |
| 难度分组表 | easy / medium / hard 上的 SI-SDRi |
| enrollment 鲁棒性表 | 1s / 3s / 5s，clean / noisy |
| 语谱图案例 | mixture、target、baseline output、ours output |
| 复杂度分布图 | complexity score histogram |
| 失败案例图 | 动态早退为什么掉指标 |

优先级：必做。该模块不一定提升模型，但能显著提升报告质量。

## 3. 当前推荐实验顺序

1. 实现 EGSP，可开关，不破坏当前主结果。
2. 用 5-block distilled checkpoint 先做 EGSP 推理探针，判断是否值得训练。
3. 若 EGSP 推理探针有效，再训练 `5-block distilled + EGSP`。
4. 生成 difficulty-aware evaluation、Pareto 图、训练曲线和语谱图。
5. 若时间允许，再做 depth-aware curriculum distillation，解决动态早退性能崩溃问题。

## 4. 当前可写结论

当前已经形成的可靠结论：

- 5-block distilled TF-GridNet-Lite 是最稳的主模型。
- 该模型以极小 SI-SDRi 损失换来明显效率提升。
- 未经 depth-aware 训练的动态早退会显著损害性能。

下一步重点：

- 用 EGSP 补强数据预处理与时频前端创新。
- 用可解释性评估补强系统完整度。
- 用 depth-aware distillation 作为动态推理的后续改进方向。

## 5. 已执行实验与最终采用方案

按照上述优先级，当前已经完成以下实验：

| 模块 | 已执行内容 | 结论 |
|---|---|---|
| 分离主干 | 继续训练 5-block distilled TF-GridNet-Lite 至早停 | 指标接近 baseline，效率明显更好，作为稳定主模型 |
| 数据预处理 / 频谱前端 | 实现 EGSP，并做 strength sweep、selected test、sanity check | 在 MiniLibriMix 设置下显著提升 SI-SDRi、SDR、SIR，作为最终 Ours 的核心前端 |
| 时频编码 | 实现 multi-resolution spectral loss 并进行真实数据 fine-tune | 可稳定训练，但当前训练预算下不超过 Ours，作为消融验证 |
| 条件模块 | 实现 speaker-aware gated fusion 并进行真实数据 fine-tune | 模块可工作，但额外参数没有带来超过 Ours 的收益 |
| 条件融合策略 | 实现 dynamic sparse fusion 并在 test set 验证 | 基本保持 Ours 指标，验证了条件注入可按复杂度调度 |
| 训练 / 推理控制 | 实现 depth-aware active-depth training 并测试 static/dynamic 推理 | dynamic 推理显著优于未训练早退，可作为 Ours-Fast 效率型变体 |
| enrollment 鲁棒性 | 测试 correct / shuffled / 1s enrollment | EGSP 同时包含目标频谱引导和通用频谱增强效应 |
| 评估与可解释性 | 生成主结果、效率、Pareto、训练曲线、难度分组、复杂度散点、语谱图案例 | 已满足报告展示和指标可信性检查需求 |

最终建议在报告中采用的系统方案为：

```text
真实 MiniLibriMix 数据
        +
Enrollment-Guided Spectral Preprocessing
        +
USEF-style speaker conditioning
        +
Sparse condition fusion
        +
5-block distilled TF-GridNet-Lite
        +
difficulty / robustness / Pareto / spectrogram interpretability
```

核心结果如下：

| Method | SI-SDRi | SDR | SIR | SAR | Params | MACs | RTF wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.1591 | 0.0522 | 0.0942 | 24.3909 | 288,818 | 10.94G | 0.002612 |
| 5-block distilled | 0.1560 | 0.0492 | 0.0899 | 24.7468 | 168,562 | 6.34G | 0.001831 |
| Ours | 0.4980 | 0.3910 | 0.4636 | 22.3793 | 168,562 | 6.34G | 0.001810 |
| Ours-Fast | 0.4554 | 0.3484 | 0.4035 | 23.2346 | 168,562 | 5.07G | 0.001650 |

5-block distilled 是最稳妥的效率创新：相比 baseline，参数量减少约 41.6%，MACs 减少约 42.0%，RTF wall 减少约 29.9%，SI-SDRi 只下降 0.0031。EGSP 是最有展示价值的系统前端创新：它不增加可训练参数和主干 MACs，但在当前数据设置下显著提升分离指标。

最终写作口径：

1. 最终质量方法统一命名为 `Ours`，即 EGSP + 5-block distilled TF-GridNet-Lite。
2. 效率型部署命名为 `Ours-Fast`，即加入 depth-aware dynamic inference。
3. gated fusion、multi-resolution spectral loss、dynamic sparse fusion 作为系统级消融验证，证明这些模块已经实现和测试，但最终组合以实测最优为准。

因此，最终主叙事应是：**以 5-block distilled 模型作为可靠效率底座，以 EGSP 作为核心频谱注意前端，以 depth-aware inference 给出效率型变体，并用完整消融与可解释性展示系统创新。**

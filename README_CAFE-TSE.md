# CAFE-TSE：基于课程学习与频域复杂度感知的高效目标说话人提取系统

> **CAFE-TSE = Curriculum-Aware Frequency-Efficient Target Speaker Extraction**  
> 面向鸡尾酒会问题的目标说话人提取系统：给定一段多人混合语音和目标说话人的参考语音，系统输出目标说话人的干净语音。

本项目面向《认知与计算》课程报告第二部分“鸡尾酒会问题模拟：基于听觉计算的多说话者语音分离系统”。系统重点模拟人类听觉注意中的三个关键机制：

- **选择性注意**：通过目标说话人参考语音 `enrollment.wav` 指定“我要听谁”；
- **特征绑定**：通过 USEF-style cross-attention 将目标说话人的声学特征绑定到混合语音中的对应声源；
- **注意资源分配**：通过 curriculum learning 与 frequency-aware dynamic inference，让模型从简单场景逐步学习复杂场景，并在推理时按语音复杂度动态分配计算量。

---

## 1. 项目目标

输入：

```text
mixture.wav              # 多人混合语音
enrollment.wav           # 目标说话人的参考语音
```

输出：

```text
target_est.wav           # 提取出的目标说话人语音
```

核心目标：

1. 在多人语音混合场景中提取指定说话人的声音；
2. 在保证 SI-SDR / SDR / SIR / SAR 尽量不明显下降的情况下，降低参数量、FLOPs 和 RTF；
3. 通过课程学习从 easy → medium → hard 逐步训练模型；
4. 通过频域复杂度感知的动态推理，对简单片段少算，对复杂片段完整推理。

---

## 2. 方法总览

### 2.1 整体结构

```text
Target enrollment speech
        │
        ▼
USEF-style target feature extractor
        │
        ▼
Target speaker condition
        │
        ├───────────────────────────────┐
        │                               │
Mixture waveform                         │
        │                               │
        ▼                               │
STFT / VAD / spectral entropy            │
        │                               │
        ▼                               │
Complexity-aware dynamic router          │
        │                               │
        ▼                               │
TF-GridNet-Lite separator  ◄─────────────┘
        │
        ▼
iSTFT
        │
        ▼
Estimated target speech
```

### 2.2 模型组成

```text
CAFE-TSE
├── USEF-style Target Conditioning
│   └── 从 enrollment 中提取目标说话人条件特征
│
├── TF-GridNet-Lite Separator
│   └── 轻量化时频域分离主干
│
├── Sparse USEF Fusion
│   └── 只在部分 block 注入目标条件，减少 cross-attention 开销
│
├── Curriculum Learning
│   └── 按 easy / medium / hard 难度逐步训练
│
└── Frequency-aware Dynamic Inference
    └── 根据 VAD、频谱熵、能量变化等指标选择浅层路径或完整路径
```

---

## 3. 创新点设计

### 3.1 TF-GridNet-Lite

在 TF-GridNet 主干基础上进行轻量化：

| 参数 | Baseline | Lite |
|---|---:|---:|
| `n_layers` | 6 | 3 或 4 |
| `emb_dim` | 48 | 32 |
| `lstm_hidden_units` | 192 | 128 |
| `attn_n_head` | 4 | 2 |

目标：

- 降低参数量；
- 降低 FLOPs / MACs；
- 降低 RTF；
- 保持目标说话人提取性能尽量不明显下降。

### 3.2 Sparse USEF Fusion

普通做法可能在多个 block 中反复注入目标说话人条件。本文采用稀疏条件融合：

```text
原始方式：
block 1: condition
block 2: condition
block 3: condition
block 4: condition

Sparse USEF Fusion：
block 1: condition
block 2: no condition
block 3: condition
block 4: no condition
```

优势：

- 保留目标说话人注意线索；
- 减少 cross-attention 调用次数；
- 降低推理开销。

### 3.3 Curriculum Learning

按照难度从低到高组织训练数据：

| Stage | 难度 | 数据设置 |
|---|---|---|
| Stage 1 | Easy | 2 人，SIR=+5 dB，clean，低重叠，enrollment=5s |
| Stage 2 | Medium | 2 人，SIR=0 dB，SNR=20/10 dB，中重叠，enrollment=3s |
| Stage 3 | Hard | 3 人，SIR=-5 dB，SNR=10/0 dB，高重叠，同性别干扰，enrollment=1s/noisy |
| Stage 4 | Mixed | easy + medium + hard 混合微调 |

训练思想：

```text
easy → medium → hard → mixed fine-tune
```

### 3.4 Frequency-aware Dynamic Inference

推理前先计算语音复杂度：

```text
complexity_score =
    a × spectral_entropy
  + b × VAD_ratio
  + c × energy_variance
  + d × spectral_flatness
```

根据复杂度选择路径：

```text
low complexity  → shallow path
high complexity → full path
```

预期效果：

- easy 样本大幅减少计算；
- hard 样本仍走完整模型，尽量保证效果；
- 平均 RTF 降低。

---

## 4. 推荐项目结构

```text
CAFE-TSE/
├── README.md
├── requirements.txt
├── configs/
│   ├── base_usef_tfgridnet.yaml
│   ├── cafe_tse_lite.yaml
│   ├── cafe_tse_curriculum.yaml
│   └── cafe_tse_dynamic.yaml
│
├── data/
│   ├── raw/
│   │   ├── LibriSpeech/
│   │   └── LibriMix/
│   ├── processed/
│   │   ├── train/
│   │   │   ├── easy/
│   │   │   ├── medium/
│   │   │   └── hard/
│   │   ├── valid/
│   │   └── test/
│   └── metadata/
│       ├── train_manifest.csv
│       ├── valid_manifest.csv
│       ├── test_manifest.csv
│       └── curriculum_manifest.csv
│
├── scripts/
│   ├── prepare_librimix.py
│   ├── build_enrollment.py
│   ├── build_curriculum_manifest.py
│   ├── compute_complexity_features.py
│   ├── run_train.sh
│   ├── run_eval.sh
│   └── run_infer.sh
│
├── src/
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── librimix_tse_dataset.py
│   │   ├── curriculum_sampler.py
│   │   └── collate_fn.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── stft.py
│   │   ├── vad.py
│   │   ├── complexity.py
│   │   └── augmentation.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cafe_tse.py
│   │   ├── usef_condition.py
│   │   ├── tfgridnet_lite.py
│   │   ├── sparse_fusion.py
│   │   └── dynamic_router.py
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── sisdr_loss.py
│   │   └── speaker_similarity_loss.py
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── separation_metrics.py
│   │   ├── efficiency_metrics.py
│   │   └── speaker_metrics.py
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── scheduler.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── audio_io.py
│       ├── logger.py
│       ├── seed.py
│       └── checkpoint.py
│
├── experiments/
│   ├── exp01_baseline_usef_tfgridnet/
│   ├── exp02_tfgridnet_lite/
│   ├── exp03_curriculum/
│   ├── exp04_dynamic_inference/
│   └── exp05_ablation/
│
└── results/
    ├── tables/
    ├── figures/
    └── audio_samples/
```

---

## 5. 环境配置

### 5.1 创建 Conda 环境

```bash
conda create -n cafe-tse python=3.10 -y
conda activate cafe-tse
```

### 5.2 安装 PyTorch

根据你的 CUDA 版本选择安装命令。例如 CUDA 11.8：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

如果没有 GPU，只做小规模测试：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5.3 安装依赖

```bash
pip install -r requirements.txt
```

建议 `requirements.txt` 包含：

```text
numpy
scipy
pandas
soundfile
librosa
tqdm
yamlargparse
omegaconf
pesq
pystoi
mir_eval
asteroid-filterbanks
speechbrain
matplotlib
seaborn
```

如果 `pesq` 安装失败，可以先注释掉，核心实验不依赖它。

---

## 6. 数据集准备

### 6.1 推荐数据集

主数据集：

```text
LibriMix / Libri2Mix / Libri3Mix
```

推荐先做小规模课程实验：

```text
Libri2Mix clean-100
Libri2Mix noisy-100
```

可选扩展：

```text
Libri3Mix
WHAMR! noisy-reverberant test set
```

### 6.2 数据准备命令

如果已经下载好 LibriMix：

```bash
python scripts/prepare_librimix.py \
  --librimix_root data/raw/LibriMix \
  --output_root data/processed \
  --sample_rate 8000 \
  --duration 4.0
```

如果暂时没有完整 LibriMix，可以先从 LibriSpeech 构造课程版小数据集：

```bash
python scripts/prepare_librimix.py \
  --librispeech_root data/raw/LibriSpeech \
  --output_root data/processed \
  --sample_rate 8000 \
  --duration 4.0 \
  --num_train 10000 \
  --num_valid 1000 \
  --num_test 1000
```

### 6.3 构造 enrollment 参考语音

```bash
python scripts/build_enrollment.py \
  --metadata data/metadata/train_manifest.csv \
  --speech_root data/raw/LibriSpeech \
  --output_dir data/processed/enrollment \
  --lengths 1 3 5 \
  --sample_rate 8000
```

### 6.4 计算复杂度特征

```bash
python scripts/compute_complexity_features.py \
  --manifest data/metadata/train_manifest.csv \
  --output data/metadata/train_complexity.csv \
  --sample_rate 8000 \
  --n_fft 512 \
  --hop_length 128
```

### 6.5 构造 curriculum manifest

```bash
python scripts/build_curriculum_manifest.py \
  --input data/metadata/train_complexity.csv \
  --output data/metadata/curriculum_manifest.csv \
  --easy_sir 5 \
  --medium_sir 0 \
  --hard_sir -5 \
  --use_overlap_ratio \
  --use_enrollment_quality
```

生成后应得到：

```text
data/metadata/curriculum_manifest.csv
```

其中包含：

```text
mixture_path,target_path,enrollment_path,num_speakers,sir,snr,overlap_ratio,enrollment_length,enrollment_noise,difficulty,complexity_score
```

---

## 7. 配置文件说明

### 7.1 `configs/base_usef_tfgridnet.yaml`

用于 baseline：USEF-TFGridNet。

```yaml
experiment_name: base_usef_tfgridnet
sample_rate: 8000
segment: 4.0

model:
  name: usef_tfgridnet
  n_layers: 6
  emb_dim: 48
  lstm_hidden_units: 192
  attn_n_head: 4
  sparse_fusion: false
  dynamic_inference: false

training:
  batch_size: 4
  epochs: 50
  lr: 0.0003
  loss: sisdr
  curriculum: false
```

### 7.2 `configs/cafe_tse_lite.yaml`

用于轻量主干实验。

```yaml
experiment_name: cafe_tse_lite
sample_rate: 8000
segment: 4.0

model:
  name: cafe_tse
  n_layers: 4
  emb_dim: 32
  lstm_hidden_units: 128
  attn_n_head: 2
  sparse_fusion: true
  fusion_layers: [0, 2]
  dynamic_inference: false

training:
  batch_size: 6
  epochs: 50
  lr: 0.0003
  loss: sisdr
  curriculum: false
```

### 7.3 `configs/cafe_tse_curriculum.yaml`

用于 curriculum learning。

```yaml
experiment_name: cafe_tse_curriculum
sample_rate: 8000
segment: 4.0

model:
  name: cafe_tse
  n_layers: 4
  emb_dim: 32
  lstm_hidden_units: 128
  attn_n_head: 2
  sparse_fusion: true
  fusion_layers: [0, 2]
  dynamic_inference: false

training:
  batch_size: 6
  epochs: 50
  lr: 0.0003
  loss: sisdr
  curriculum: true
  curriculum_schedule:
    easy_epochs: 10
    medium_epochs: 20
    hard_epochs: 35
    mixed_epochs: 50
```

### 7.4 `configs/cafe_tse_dynamic.yaml`

用于最终 CAFE-TSE。

```yaml
experiment_name: cafe_tse_dynamic
sample_rate: 8000
segment: 4.0

model:
  name: cafe_tse
  n_layers: 4
  emb_dim: 32
  lstm_hidden_units: 128
  attn_n_head: 2
  sparse_fusion: true
  fusion_layers: [0, 2]
  dynamic_inference: true
  dynamic_router:
    easy_threshold: 0.35
    hard_threshold: 0.65
    shallow_layers: 2

training:
  batch_size: 6
  epochs: 50
  lr: 0.0003
  loss: sisdr
  curriculum: true
```

---

## 8. 实验步骤与命令

### 8.1 Step 1：检查数据

```bash
python scripts/check_data.py \
  --manifest data/metadata/train_manifest.csv
```

预期输出：

```text
Number of train samples: xxxx
Number of valid samples: xxxx
Number of test samples: xxxx
Found mixture / target / enrollment files: OK
```

### 8.2 Step 2：训练 baseline USEF-TFGridNet

```bash
python train.py \
  --config configs/base_usef_tfgridnet.yaml \
  --train_manifest data/metadata/train_manifest.csv \
  --valid_manifest data/metadata/valid_manifest.csv \
  --exp_dir experiments/exp01_baseline_usef_tfgridnet
```

### 8.3 Step 3：训练 USEF-TFGridNet-Lite

```bash
python train.py \
  --config configs/cafe_tse_lite.yaml \
  --train_manifest data/metadata/train_manifest.csv \
  --valid_manifest data/metadata/valid_manifest.csv \
  --exp_dir experiments/exp02_tfgridnet_lite
```

### 8.4 Step 4：训练 Curriculum 版本

```bash
python train.py \
  --config configs/cafe_tse_curriculum.yaml \
  --train_manifest data/metadata/curriculum_manifest.csv \
  --valid_manifest data/metadata/valid_manifest.csv \
  --exp_dir experiments/exp03_curriculum
```

### 8.5 Step 5：训练最终 CAFE-TSE

```bash
python train.py \
  --config configs/cafe_tse_dynamic.yaml \
  --train_manifest data/metadata/curriculum_manifest.csv \
  --valid_manifest data/metadata/valid_manifest.csv \
  --exp_dir experiments/exp04_dynamic_inference
```

### 8.6 Step 6：评估分离质量

```bash
python evaluate.py \
  --config configs/cafe_tse_dynamic.yaml \
  --test_manifest data/metadata/test_manifest.csv \
  --checkpoint experiments/exp04_dynamic_inference/best.ckpt \
  --output_dir results/tables/cafe_tse_dynamic
```

输出指标：

```text
SI-SDR
SI-SDRi
SDR
SIR
SAR
STOI
PESQ optional
```

### 8.7 Step 7：评估效率

```bash
python evaluate_efficiency.py \
  --config configs/cafe_tse_dynamic.yaml \
  --test_manifest data/metadata/test_manifest.csv \
  --checkpoint experiments/exp04_dynamic_inference/best.ckpt \
  --output results/tables/efficiency_cafe_tse.csv \
  --num_samples 200
```

输出指标：

```text
Params
FLOPs / MACs
RTF
Inference time
Peak GPU memory
Average active layers
Skip ratio
```

### 8.8 Step 8：按难度分组评估

```bash
python evaluate_by_difficulty.py \
  --config configs/cafe_tse_dynamic.yaml \
  --test_manifest data/metadata/test_manifest.csv \
  --checkpoint experiments/exp04_dynamic_inference/best.ckpt \
  --output results/tables/difficulty_results.csv
```

结果应分为：

```text
easy
medium
hard
```

### 8.9 Step 9：Enrollment 鲁棒性实验

```bash
python evaluate_enrollment_robustness.py \
  --config configs/cafe_tse_dynamic.yaml \
  --test_manifest data/metadata/test_manifest.csv \
  --checkpoint experiments/exp04_dynamic_inference/best.ckpt \
  --lengths 1 3 5 \
  --noise_levels clean 20 10 \
  --output results/tables/enrollment_robustness.csv
```

### 8.10 Step 10：单条语音推理

```bash
python infer.py \
  --config configs/cafe_tse_dynamic.yaml \
  --checkpoint experiments/exp04_dynamic_inference/best.ckpt \
  --mixture examples/mixture.wav \
  --enrollment examples/enrollment.wav \
  --output results/audio_samples/target_est.wav
```

---

## 9. 消融实验命令

### 9.1 去掉 Curriculum Learning

```bash
python train.py \
  --config configs/cafe_tse_lite.yaml \
  --train_manifest data/metadata/train_manifest.csv \
  --valid_manifest data/metadata/valid_manifest.csv \
  --exp_dir experiments/exp05_ablation/w_o_curriculum
```

### 9.2 去掉 Dynamic Inference

```bash
python train.py \
  --config configs/cafe_tse_curriculum.yaml \
  --train_manifest data/metadata/curriculum_manifest.csv \
  --valid_manifest data/metadata/valid_manifest.csv \
  --exp_dir experiments/exp05_ablation/w_o_dynamic
```

### 9.3 去掉 Sparse USEF Fusion

```bash
python train.py \
  --config configs/cafe_tse_curriculum.yaml \
  --override model.sparse_fusion=false \
  --train_manifest data/metadata/curriculum_manifest.csv \
  --valid_manifest data/metadata/valid_manifest.csv \
  --exp_dir experiments/exp05_ablation/w_o_sparse_fusion
```

### 9.4 去掉 Lite 主干

```bash
python train.py \
  --config configs/base_usef_tfgridnet.yaml \
  --override training.curriculum=true \
  --train_manifest data/metadata/curriculum_manifest.csv \
  --valid_manifest data/metadata/valid_manifest.csv \
  --exp_dir experiments/exp05_ablation/w_o_lite
```

### 9.5 随机课程顺序对照

```bash
python train.py \
  --config configs/cafe_tse_curriculum.yaml \
  --override training.curriculum_mode=random \
  --train_manifest data/metadata/curriculum_manifest.csv \
  --valid_manifest data/metadata/valid_manifest.csv \
  --exp_dir experiments/exp05_ablation/random_curriculum
```

---

## 10. 推荐实验表格

### 10.1 主结果表

| Method | SI-SDRi ↑ | SDR ↑ | SIR ↑ | SAR ↑ | Params ↓ | FLOPs ↓ | RTF ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mixture | - | - | - | - | 0 | 0 | 0 |
| USEF-TFGridNet |  |  |  |  |  |  |  |
| USEF-TFGridNet-Lite |  |  |  |  |  |  |  |
| CAFE-TSE w/o Dynamic |  |  |  |  |  |  |  |
| CAFE-TSE |  |  |  |  |  |  |  |

### 10.2 消融实验表

| Method | SI-SDRi ↑ | RTF ↓ | Skip Ratio ↑ | Conclusion |
|---|---:|---:|---:|---|
| Full CAFE-TSE |  |  |  | 完整方法 |
| w/o Curriculum |  |  |  | 验证课程学习 |
| w/o Dynamic |  |  |  | 验证动态推理 |
| w/o Sparse Fusion |  |  |  | 验证稀疏条件融合 |
| w/o Lite |  |  |  | 验证轻量化主干 |

### 10.3 难度分组表

| Difficulty | Method | SI-SDRi ↑ | RTF ↓ | Skip Ratio ↑ |
|---|---|---:|---:|---:|
| Easy | CAFE-TSE |  |  |  |
| Medium | CAFE-TSE |  |  |  |
| Hard | CAFE-TSE |  |  |  |

### 10.4 Enrollment 鲁棒性表

| Enrollment Length | Enrollment Noise | SI-SDRi ↑ | SIR ↑ | Speaker Similarity ↑ |
|---:|---|---:|---:|---:|
| 1s | clean |  |  |  |
| 3s | clean |  |  |  |
| 5s | clean |  |  |  |
| 3s | 20 dB |  |  |  |
| 3s | 10 dB |  |  |  |

---

## 11. 预期结论

预期实验现象：

1. `USEF-TFGridNet-Lite` 相比 `USEF-TFGridNet` 参数量和 RTF 明显降低，但 SI-SDRi 可能小幅下降；
2. 加入 curriculum learning 后，hard 样本上的收敛更稳定，模型对低 SIR / 高重叠样本更鲁棒；
3. 加入 dynamic inference 后，easy 样本上的 RTF 明显降低，整体平均推理时间下降；
4. hard 样本大多走 full path，因此效果不会明显下降；
5. enrollment 越短或越吵，目标说话人绑定越困难，speaker similarity 和 SI-SDRi 下降。

最终希望证明：

```text
CAFE-TSE 能够在保持目标说话人提取效果基本稳定的同时，降低平均推理开销，并且更好地对应人类听觉注意中的选择性注意、特征绑定和注意资源动态分配机制。
```

---

## 12. 最小可行实验路线

如果时间有限，优先完成以下内容：

```text
1. 数据准备：Libri2Mix clean 小规模子集
2. Baseline：USEF-TFGridNet 或简化 USEF-TFGridNet
3. Lite：USEF-TFGridNet-Lite
4. Curriculum：easy → medium → hard 训练
5. Dynamic：基于 complexity score 的浅层/完整路径选择
6. 评估：SI-SDRi / SDR / SIR / SAR / RTF / Params
7. 消融：w/o curriculum、w/o dynamic、w/o sparse fusion
```

最低需要跑出的表格：

```text
主结果表 + 消融实验表 + 难度分组表
```

---

## 13. 常见问题

### Q1：如果 USEF-TFGridNet 太难跑怎么办？

先实现简化版：

```text
STFT → TF-GridNet-Lite → target conditioning → iSTFT
```

或者先用 Conv-TasNet / SepFormer 作为 baseline，再逐步替换成 TF-GridNet-Lite。

### Q2：如果 dynamic inference 导致效果下降怎么办？

降低跳过强度：

```text
只让 very easy 样本走 shallow path
medium 和 hard 全部走 full path
```

### Q3：如果 curriculum learning 没提升怎么办？

仍然可以作为数据难度分层分析。报告中可以说明：

```text
课程学习对 hard 样本有一定帮助，但对整体平均指标提升有限。
```

### Q4：如果 PESQ 安装失败怎么办？

可以不使用 PESQ，主指标保留：

```text
SI-SDRi / SDR / SIR / SAR / RTF / Params
```

---

## 14. 一句话总结

CAFE-TSE 是一个面向鸡尾酒会问题的高效目标说话人提取系统。它以 USEF-style 目标说话人条件建模和 TF-GridNet-Lite 时频域分离主干为基础，通过课程学习从易到难训练模型，并利用频域复杂度感知的动态推理机制在不同难度片段上自适应分配计算资源，从而兼顾语音提取效果与推理效率。

# CAFE-TSE: Curriculum-Aware Frequency-Efficient Target Speaker Extraction

> 基于课程学习与频域复杂度感知的高效目标说话人提取系统。  
> 目标：在多人混合语音中，根据目标说话人的 enrollment/reference 语音，提取该特定说话人的声音，并同时分析分离质量与推理效率。

---

## 0. 给 Codex 的总目标

请根据本 README 构建一个**完整可运行**的 PyTorch 项目，而不是只写一个玩具 demo。项目应能完成以下闭环：

```text
数据准备 / manifest 构建
  ↓
目标说话人 enrollment 构造
  ↓
复杂度特征计算与 curriculum difficulty 标注
  ↓
训练 USEF-style + TF-GridNet-Lite 目标说话人提取模型
  ↓
训练 CAFE-TSE：curriculum learning + sparse fusion + dynamic inference
  ↓
评估 SI-SDRi / SDR / SIR / SAR / RTF / Params / Skip Ratio
  ↓
导出实验表格、音频样例和可写进课程报告的结果
```

不要把本项目写成“随机生成数据 + 简单网络”的最小包。可以提供 smoke test / toy data 用于验证代码，但主流程必须面向 LibriMix / Libri2Mix 数据集。

---

## 1. 课程任务对应关系

课程第二部分要求实现“鸡尾酒会情景下的多说话者语音分离系统”，并使模型能够在多人交谈中“跟踪并集中听取某个特定说话者的声音”。因此本项目任务定义为：

```text
Input:
  mixture.wav             # 多说话者混合语音
  enrollment.wav          # 目标说话人的参考语音

Output:
  estimated_target.wav    # 从 mixture 中提取出的目标说话人语音
```

评分标准强调：

- 模型设计与实现：架构清晰、模拟声源定位/特征绑定/选择性注意、代码完整、有创新性。
- 实验与验证：不同信噪比、说话者数量等条件，SDR/SIR/SAR 等指标，结果分析充分。
- 理论分析：讨论与人类听觉行为的异同，计算效率、泛化能力和局限性。

CAFE-TSE 对应关系：

| 作业要求 | 本项目实现 |
|---|---|
| 多说话者语音分离 | TF-GridNet-Lite 时频域分离主干 |
| 特定说话者跟踪 | enrollment/reference 目标说话人输入 |
| 特征绑定 | USEF-style cross-attention 条件融合 |
| 选择性注意 | 根据 target condition 提取目标说话人 |
| 注意资源分配 | frequency complexity-aware dynamic inference |
| 不同信噪比/说话人数实验 | Libri2Mix / Libri3Mix + SIR/SNR/difficulty 分层 |
| SDR/SIR/SAR 指标 | museval / fast-bss-eval / torchmetrics-style metrics |
| 计算效率讨论 | Params / MACs / RTF / active blocks / skip ratio |

---

## 2. 方法总览

本项目命名为：

```text
CAFE-TSE
= Curriculum-Aware Frequency-Efficient Target Speaker Extraction
```

核心组成：

```text
CAFE-TSE
= USEF-style target conditioning
+ TF-GridNet-Lite separator
+ Sparse USEF Fusion
+ Difficulty-aware Curriculum Learning
+ Frequency Complexity-aware Dynamic Inference
```

整体 pipeline：

```text
                       ┌────────────────────────────┐
                       │ target enrollment waveform │
                       └─────────────┬──────────────┘
                                     ↓
                              STFT / encoder
                                     ↓
                    USEF-style target feature extractor
                          cross-attention / FiLM
                                     ↓
                             target condition
                                     ↓
┌───────────────────┐       ┌───────────────────────┐
│ mixture waveform  │  ───▶ │ STFT complex spectrum │
└───────────────────┘       └───────────┬───────────┘
                                        ↓
                        complexity feature extractor
              VAD ratio / spectral entropy / energy variance
                                        ↓
                              dynamic route decision
                       shallow path / lite path / full path
                                        ↓
                         TF-GridNet-Lite separator blocks
                                        ↓
                         target complex spectrum estimate
                                        ↓
                                      iSTFT
                                        ↓
                           estimated target waveform
```

---

## 3. 强边界限制：必须遵守

这部分是 Codex 构建代码时的**硬约束**。

### 3.1 不要做的事

1. **不要引入 Mamba / mamba-ssm / causal-conv1d** 作为主路径依赖。Mamba 可写在报告未来工作里，但不要作为代码核心。
2. **不要直接复制 USEF-TSE、ESPnet、Asteroid 的大段源码到本项目 src/**。允许 clone 到 `third_party/` 作为参考或可选 baseline，但本项目核心实现必须在 `src/cafe_tse/` 下完成。
3. **不要依赖绝对路径**，所有路径从 config 或 CLI 参数读取。
4. **不要把数据下载、训练、评估混在一个脚本里**。必须拆成清晰 CLI。
5. **不要只实现 synthetic toy data**。Toy data 只能用于 smoke test，主数据流程必须支持 LibriMix manifest。
6. **不要默默跳过指标**。如果 SDR/SIR/SAR 依赖缺失，应报出明确 warning，并至少输出 SI-SDR / SI-SDRi / RTF / Params。
7. **不要让训练脚本只支持单一实验**。必须支持 baseline、lite、curriculum、dynamic 和 ablation。
8. **不要在 README 中承诺必须跑完整 430GB LibriMix**。默认实验应支持 small subset。
9. **不要把第三方 checkpoint 当成唯一可运行条件**。项目要能从头训练 small subset。
10. **不要把数据集许可证文件删除或混入提交**。第三方数据与权重只放在 data/ 和 third_party/，不要纳入 git。

### 3.2 必须实现的 harness

项目根目录必须提供以下 harness 文件：

```text
scripts/harness_smoke.sh
scripts/harness_toy_train_eval.sh
scripts/harness_librimix_small.sh
```

含义：

- `harness_smoke.sh`：不依赖真实数据，生成 8 条 toy wav，跑通 manifest、训练 1 epoch、评估、推理。
- `harness_toy_train_eval.sh`：使用 synthetic toy dataset 训练 2-3 epoch，验证 loss 下降、输出音频和 metrics.csv。
- `harness_librimix_small.sh`：假设 LibriMix 已经生成或已给路径，抽取 small subset，跑通完整实验流程。

最低验收命令：

```bash
bash scripts/harness_smoke.sh
```

该命令必须完成：

```text
1. 创建 data/toy/
2. 生成 toy manifest
3. 计算 complexity features
4. 训练 tiny CAFE-TSE 1 epoch
5. 评估并生成 results/smoke/metrics.csv
6. 推理生成 results/smoke/audio/estimated_0.wav
7. 运行 pytest tests/test_*.py
```

---

## 4. 官方源码与数据集命令

### 4.1 推荐 clone 的参考仓库

所有第三方仓库放到 `third_party/`。不要把它们的源码直接改成项目主代码。

```bash
mkdir -p third_party

# USEF-TSE 官方实现：目标说话人提取框架参考
# 官方 README 中提供 train.sh / eval.sh / checkpoints 说明
# 仅作为参考或 baseline，不要直接覆盖本项目 src/cafe_tse/
git clone https://github.com/ZBang/USEF-TSE.git third_party/USEF-TSE

# LibriMix 数据集生成脚本：用于生成 Libri2Mix / Libri3Mix
git clone https://github.com/JorisCos/LibriMix.git third_party/LibriMix

# ESPnet：参考 TFGridNet separator 的工程实现
# 只作参考，不要求安装完整 ESPnet
git clone --depth 1 https://github.com/espnet/espnet.git third_party/espnet

# Asteroid：可选 baseline / metrics / LibriMix recipe 参考
git clone --depth 1 https://github.com/asteroid-team/asteroid.git third_party/asteroid
```

### 4.2 LibriMix 生成命令

官方 LibriMix README 的基本生成方式为：

```bash
cd third_party/LibriMix
conda install -c conda-forge sox -y
./generate_librimix.sh /path/to/storage_dir
```

完整默认生成会占用较大空间。课程实验建议只生成 small subset 或只使用 Libri2Mix clean/noisy 16k min 版本。Codex 需要实现本项目自己的 manifest 构建脚本，能够从已有 LibriMix 目录中读取数据，而不是强制重新生成全部数据。

建议目录：

```text
data/raw/LibriMix/
  Libri2Mix/
    wav16k/
      min/
        train-100/
        dev/
        test/
```

若从 OpenSLR 手动下载 LibriSpeech，可参考：

```bash
mkdir -p data/downloads data/raw/LibriSpeech
cd data/downloads

# train-clean-100: about 6.3G
wget -c https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget -c https://www.openslr.org/resources/12/dev-clean.tar.gz
wget -c https://www.openslr.org/resources/12/test-clean.tar.gz

tar -xzf train-clean-100.tar.gz -C ../raw/LibriSpeech
tar -xzf dev-clean.tar.gz -C ../raw/LibriSpeech
tar -xzf test-clean.tar.gz -C ../raw/LibriSpeech
```

注意：LibriMix 的官方脚本通常会自动处理 LibriSpeech / WHAM noise 下载与生成流程。优先使用 `third_party/LibriMix/generate_librimix.sh`，手动下载只作为网络不稳定时的备选。

---

## 5. 环境配置

推荐 Linux + CUDA + Python 3.10。

```bash
conda create -n cafe-tse python=3.10 -y
conda activate cafe-tse

# 根据自己的 CUDA 版本安装 PyTorch，例如 CUDA 12.1：
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

`requirements.txt` 至少包含：

```text
torch
torchaudio
numpy
scipy
pandas
soundfile
librosa
pyyaml
tqdm
mir_eval
museval
fast-bss-eval
pesq
pystoi
thop
pytest
rich
```

如果 `pesq` 安装失败，不要让整个项目不可运行。评估脚本应把 PESQ 标记为 optional。

---

## 6. 目标项目结构

Codex 必须按以下结构实现：

```text
CAFE-TSE/
  README.md
  requirements.txt
  pyproject.toml
  .gitignore

  configs/
    base_usef_tfgridnet.yaml
    cafe_tse_lite.yaml
    cafe_tse_curriculum.yaml
    cafe_tse_dynamic.yaml
    ablation_no_curriculum.yaml
    ablation_no_dynamic.yaml
    ablation_no_sparse_fusion.yaml
    smoke_tiny.yaml

  scripts/
    harness_smoke.sh
    harness_toy_train_eval.sh
    harness_librimix_small.sh
    prepare_third_party.sh
    run_train.sh
    run_eval.sh
    run_infer.sh

  data/
    raw/
    processed/
    metadata/
    toy/

  src/
    cafe_tse/
      __init__.py

      cli/
        __init__.py
        prepare_toy_data.py
        prepare_librimix_manifest.py
        build_enrollment.py
        compute_complexity_manifest.py
        train.py
        evaluate.py
        infer.py
        summarize_results.py

      datasets/
        __init__.py
        tse_dataset.py
        curriculum_sampler.py
        collate.py

      features/
        __init__.py
        stft.py
        vad.py
        complexity.py
        augmentation.py

      models/
        __init__.py
        cafe_tse.py
        usef_condition.py
        tfgridnet_lite.py
        sparse_fusion.py
        dynamic_router.py

      losses/
        __init__.py
        sisdr.py
        spectral.py

      metrics/
        __init__.py
        separation.py
        efficiency.py
        speaker.py

      engine/
        __init__.py
        trainer.py
        evaluator.py
        checkpoint.py

      utils/
        __init__.py
        audio_io.py
        config.py
        logger.py
        seed.py
        manifest.py

  tests/
    test_complexity.py
    test_router.py
    test_dataset.py
    test_model_forward.py
    test_sisdr.py

  experiments/
    .gitkeep

  results/
    .gitkeep

  third_party/
    .gitkeep
```

---

## 7. Manifest 数据格式

所有训练、验证、测试数据统一走 CSV manifest。

`data/metadata/train_manifest.csv` 字段必须包含：

```text
utt_id
mixture_path
target_path
enrollment_path
speaker_id
split
sample_rate
duration
num_speakers
sir
snr
overlap_ratio
gender_condition
enrollment_length
enrollment_noise
difficulty
complexity_score
```

其中：

- `difficulty`: one of `easy`, `medium`, `hard`
- `complexity_score`: float, 0-1 之间最好；如果未归一化，脚本必须在训练前归一化。
- `speaker_id`: 目标说话人的 ID。
- `target_path`: ground-truth target source wav。
- `enrollment_path`: 与 target speaker 同 speaker 但不同 utterance 的参考语音；toy 数据中可用同源切片。

---

## 8. 数据准备命令

### 8.1 Toy smoke data

```bash
python -m cafe_tse.cli.prepare_toy_data \
  --out_dir data/toy \
  --num_samples 16 \
  --sample_rate 8000 \
  --duration 2.0 \
  --num_speakers 2

python -m cafe_tse.cli.compute_complexity_manifest \
  --manifest data/toy/toy_manifest.csv \
  --out_manifest data/toy/toy_manifest_complexity.csv \
  --sample_rate 8000
```

### 8.2 从 LibriMix 构建 TSE manifest

假设 LibriMix 已在：

```text
data/raw/LibriMix/Libri2Mix/wav16k/min/
```

运行：

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
```

该脚本需要输出：

```text
data/metadata/librimix/train_manifest.csv
data/metadata/librimix/valid_manifest.csv
data/metadata/librimix/test_manifest.csv
```

### 8.3 构造 enrollment

```bash
python -m cafe_tse.cli.build_enrollment \
  --manifest data/metadata/librimix/train_manifest.csv \
  --speaker_pool_root data/raw/LibriSpeech/LibriSpeech/train-clean-100 \
  --out_manifest data/metadata/librimix/train_manifest_enroll.csv \
  --enrollment_seconds 3.0 \
  --strategy different_utterance
```

如果使用 LibriMix 已包含 source 路径，也可以从同 speaker 的其他 source 中采样 enrollment。Codex 实现时要支持两种方式：

```text
strategy=different_utterance
strategy=source_crop
```

### 8.4 计算复杂度与 difficulty

```bash
python -m cafe_tse.cli.compute_complexity_manifest \
  --manifest data/metadata/librimix/train_manifest_enroll.csv \
  --out_manifest data/metadata/librimix/train_manifest_final.csv \
  --sample_rate 16000 \
  --difficulty_rule curriculum_v1
```

同样处理 valid/test：

```bash
python -m cafe_tse.cli.compute_complexity_manifest \
  --manifest data/metadata/librimix/valid_manifest.csv \
  --out_manifest data/metadata/librimix/valid_manifest_final.csv \
  --sample_rate 16000

python -m cafe_tse.cli.compute_complexity_manifest \
  --manifest data/metadata/librimix/test_manifest.csv \
  --out_manifest data/metadata/librimix/test_manifest_final.csv \
  --sample_rate 16000
```

---

## 9. 训练命令

### 9.1 Smoke tiny train

```bash
python -m cafe_tse.cli.train \
  --config configs/smoke_tiny.yaml \
  --train_manifest data/toy/toy_manifest_complexity.csv \
  --valid_manifest data/toy/toy_manifest_complexity.csv \
  --exp_dir experiments/smoke_tiny
```

### 9.2 Baseline: USEF-style TFGridNet

```bash
python -m cafe_tse.cli.train \
  --config configs/base_usef_tfgridnet.yaml \
  --train_manifest data/metadata/librimix/train_manifest_final.csv \
  --valid_manifest data/metadata/librimix/valid_manifest_final.csv \
  --exp_dir experiments/exp01_base_usef_tfgridnet
```

### 9.3 Lite backbone

```bash
python -m cafe_tse.cli.train \
  --config configs/cafe_tse_lite.yaml \
  --train_manifest data/metadata/librimix/train_manifest_final.csv \
  --valid_manifest data/metadata/librimix/valid_manifest_final.csv \
  --exp_dir experiments/exp02_cafe_tse_lite
```

### 9.4 Curriculum learning

```bash
python -m cafe_tse.cli.train \
  --config configs/cafe_tse_curriculum.yaml \
  --train_manifest data/metadata/librimix/train_manifest_final.csv \
  --valid_manifest data/metadata/librimix/valid_manifest_final.csv \
  --exp_dir experiments/exp03_cafe_tse_curriculum
```

### 9.5 Dynamic inference final model

```bash
python -m cafe_tse.cli.train \
  --config configs/cafe_tse_dynamic.yaml \
  --train_manifest data/metadata/librimix/train_manifest_final.csv \
  --valid_manifest data/metadata/librimix/valid_manifest_final.csv \
  --exp_dir experiments/exp04_cafe_tse_dynamic
```

---

## 10. 评估命令

### 10.1 单模型评估

```bash
python -m cafe_tse.cli.evaluate \
  --config configs/cafe_tse_dynamic.yaml \
  --checkpoint experiments/exp04_cafe_tse_dynamic/checkpoints/best.pt \
  --test_manifest data/metadata/librimix/test_manifest_final.csv \
  --out_dir results/exp04_cafe_tse_dynamic \
  --save_audio 20
```

输出：

```text
results/exp04_cafe_tse_dynamic/metrics.csv
results/exp04_cafe_tse_dynamic/summary.json
results/exp04_cafe_tse_dynamic/audio/*.wav
```

`metrics.csv` 至少包含：

```text
utt_id
si_sdr
si_sdri
sdr
sir
sar
stoi
pesq
rtf
route
active_blocks
complexity_score
difficulty
```

### 10.2 对比实验汇总

```bash
python -m cafe_tse.cli.summarize_results \
  --result_dirs \
    results/exp01_base_usef_tfgridnet \
    results/exp02_cafe_tse_lite \
    results/exp03_cafe_tse_curriculum \
    results/exp04_cafe_tse_dynamic \
  --out_csv results/summary_main_table.csv \
  --out_md results/summary_main_table.md
```

---

## 11. 消融实验命令

### 11.1 去掉 curriculum

```bash
python -m cafe_tse.cli.train \
  --config configs/ablation_no_curriculum.yaml \
  --train_manifest data/metadata/librimix/train_manifest_final.csv \
  --valid_manifest data/metadata/librimix/valid_manifest_final.csv \
  --exp_dir experiments/ablation_no_curriculum
```

### 11.2 去掉 dynamic inference

```bash
python -m cafe_tse.cli.train \
  --config configs/ablation_no_dynamic.yaml \
  --train_manifest data/metadata/librimix/train_manifest_final.csv \
  --valid_manifest data/metadata/librimix/valid_manifest_final.csv \
  --exp_dir experiments/ablation_no_dynamic
```

### 11.3 去掉 sparse fusion

```bash
python -m cafe_tse.cli.train \
  --config configs/ablation_no_sparse_fusion.yaml \
  --train_manifest data/metadata/librimix/train_manifest_final.csv \
  --valid_manifest data/metadata/librimix/valid_manifest_final.csv \
  --exp_dir experiments/ablation_no_sparse_fusion
```

### 11.4 批量评估 ablation

```bash
for EXP in ablation_no_curriculum ablation_no_dynamic ablation_no_sparse_fusion; do
  python -m cafe_tse.cli.evaluate \
    --config configs/${EXP}.yaml \
    --checkpoint experiments/${EXP}/checkpoints/best.pt \
    --test_manifest data/metadata/librimix/test_manifest_final.csv \
    --out_dir results/${EXP} \
    --save_audio 10
done
```

---

## 12. 推理命令

对单条 mixture + enrollment 推理：

```bash
python -m cafe_tse.cli.infer \
  --config configs/cafe_tse_dynamic.yaml \
  --checkpoint experiments/exp04_cafe_tse_dynamic/checkpoints/best.pt \
  --mixture examples/mixture.wav \
  --enrollment examples/enrollment.wav \
  --out_wav results/demo/estimated_target.wav \
  --device cuda
```

要求输出 route 信息：

```text
complexity_score: 0.62
route: full
active_blocks: 4
rtf: 0.08
saved: results/demo/estimated_target.wav
```

---

## 13. Config 约定

`configs/cafe_tse_dynamic.yaml` 示例：

```yaml
seed: 42
sample_rate: 16000
segment_seconds: 4.0
batch_size: 4
num_workers: 4
max_epochs: 50
precision: fp32

data:
  normalize_audio: true
  curriculum: true
  curriculum_schedule:
    - {until_epoch: 10, difficulties: [easy]}
    - {until_epoch: 25, difficulties: [easy, medium]}
    - {until_epoch: 40, difficulties: [medium, hard]}
    - {until_epoch: 50, difficulties: [easy, medium, hard]}

model:
  name: cafe_tse
  n_fft: 512
  hop_length: 128
  win_length: 512
  emb_dim: 32
  hidden_dim: 128
  n_blocks: 4
  n_heads: 2
  sparse_fusion_blocks: [0, 2]
  dynamic_inference: true
  shallow_blocks: 2
  full_blocks: 4
  route_threshold_easy: 0.35
  route_threshold_hard: 0.65

loss:
  si_sdr_weight: 1.0
  spectral_weight: 0.2

optim:
  name: adamw
  lr: 0.0003
  weight_decay: 0.00001
  grad_clip: 5.0

eval:
  save_audio: 20
  compute_sdr_sir_sar: true
  compute_pesq_stoi: false
```

---

## 14. 关键代码规范

下面给出关键模块的代码骨架。Codex 应根据这些骨架实现完整代码。

### 14.1 Complexity features

文件：`src/cafe_tse/features/complexity.py`

```python
from __future__ import annotations

import torch


def spectral_entropy(magnitude: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute normalized spectral entropy.

    Args:
        magnitude: Tensor with shape [..., freq, time].
    Returns:
        Tensor with shape [...].
    """
    mag = magnitude.clamp_min(eps)
    prob = mag / mag.sum(dim=-2, keepdim=True).clamp_min(eps)
    ent = -(prob * prob.log()).sum(dim=-2)
    ent = ent / torch.log(torch.tensor(magnitude.shape[-2], device=magnitude.device, dtype=magnitude.dtype))
    return ent.mean(dim=-1)


def energy_variance(wav: torch.Tensor, frame: int = 512, hop: int = 128, eps: float = 1e-8) -> torch.Tensor:
    """Frame-level energy variance normalized by mean energy."""
    frames = wav.unfold(-1, frame, hop)
    energy = frames.pow(2).mean(dim=-1)
    return energy.var(dim=-1) / energy.mean(dim=-1).clamp_min(eps)


def vad_ratio(wav: torch.Tensor, frame: int = 512, hop: int = 128, threshold: float = 1e-4) -> torch.Tensor:
    frames = wav.unfold(-1, frame, hop)
    energy = frames.pow(2).mean(dim=-1)
    return (energy > threshold).float().mean(dim=-1)


def spectral_flatness(magnitude: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mag = magnitude.clamp_min(eps)
    geo = mag.log().mean(dim=-2).exp()
    arith = mag.mean(dim=-2).clamp_min(eps)
    return (geo / arith).mean(dim=-1)


def compute_complexity_score(
    wav: torch.Tensor,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int = 128,
    weights: dict | None = None,
) -> torch.Tensor:
    """Return a 0-1-ish difficulty score for dynamic inference.

    The score is heuristic. It must be deterministic and logged for every sample.
    """
    if weights is None:
        weights = {
            "entropy": 0.35,
            "vad": 0.25,
            "energy": 0.25,
            "flatness": 0.15,
        }

    window = torch.hann_window(n_fft, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    mag = spec.abs()

    ent = spectral_entropy(mag)
    vad = vad_ratio(wav, frame=n_fft, hop=hop_length)
    evar = energy_variance(wav, frame=n_fft, hop=hop_length)
    flat = spectral_flatness(mag)

    # evar may be greater than 1; squash to stable range.
    evar = torch.tanh(evar)

    score = (
        weights["entropy"] * ent
        + weights["vad"] * vad
        + weights["energy"] * evar
        + weights["flatness"] * flat
    )
    return score.clamp(0.0, 1.0)
```

### 14.2 Dynamic router

文件：`src/cafe_tse/models/dynamic_router.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn


class DynamicRouter(nn.Module):
    """Rule-based route selector for frequency-efficient inference.

    Keep it rule-based for the course project so that the behavior is explainable.
    """

    def __init__(self, threshold_easy: float = 0.35, threshold_hard: float = 0.65):
        super().__init__()
        self.threshold_easy = threshold_easy
        self.threshold_hard = threshold_hard

    def forward(self, complexity_score: torch.Tensor) -> list[str]:
        routes = []
        for s in complexity_score.detach().cpu().tolist():
            if s < self.threshold_easy:
                routes.append("shallow")
            elif s < self.threshold_hard:
                routes.append("lite")
            else:
                routes.append("full")
        return routes

    def active_blocks(self, routes: list[str], shallow_blocks: int, lite_blocks: int, full_blocks: int) -> list[int]:
        out = []
        for route in routes:
            if route == "shallow":
                out.append(shallow_blocks)
            elif route == "lite":
                out.append(lite_blocks)
            else:
                out.append(full_blocks)
        return out
```

### 14.3 Sparse USEF Fusion

文件：`src/cafe_tse/models/sparse_fusion.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn


class SparseConditionFusion(nn.Module):
    """Inject target speaker condition only at selected separator blocks."""

    def __init__(self, dim: int, condition_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(condition_dim, dim)
        self.to_beta = nn.Linear(condition_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """FiLM-style conditioning.

        Args:
            x: [B, T, F, C]
            condition: [B, C_cond]
        Returns:
            conditioned x with same shape.
        """
        x_norm = self.norm(x)
        gamma = self.to_gamma(condition).unsqueeze(1).unsqueeze(1)
        beta = self.to_beta(condition).unsqueeze(1).unsqueeze(1)
        return x + gamma * x_norm + beta
```

### 14.4 USEF-style target condition extractor

文件：`src/cafe_tse/models/usef_condition.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn


class USEFConditionExtractor(nn.Module):
    """Embedding-free target condition extractor using cross-attention.

    Query comes from mixture features; key/value come from enrollment features.
    This follows the USEF-style idea without requiring an external speaker encoder.
    """

    def __init__(self, dim: int, n_heads: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, mix_tokens: torch.Tensor, enroll_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mix_tokens: [B, Tm, C]
            enroll_tokens: [B, Te, C]
        Returns:
            condition: [B, C]
        """
        attended, _ = self.attn(query=mix_tokens, key=enroll_tokens, value=enroll_tokens)
        attended = self.norm(attended + mix_tokens)
        condition = attended.mean(dim=1)
        return condition
```

### 14.5 CAFE-TSE forward

文件：`src/cafe_tse/models/cafe_tse.py`

```python
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from cafe_tse.features.complexity import compute_complexity_score
from cafe_tse.models.dynamic_router import DynamicRouter
from cafe_tse.models.usef_condition import USEFConditionExtractor
from cafe_tse.models.tfgridnet_lite import TFGridNetLite


@dataclass
class CafeTSEOutput:
    wav: torch.Tensor
    route: list[str]
    complexity_score: torch.Tensor
    active_blocks: list[int]
    rtf: float | None = None


class CafeTSE(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        emb_dim: int = 32,
        hidden_dim: int = 128,
        n_blocks: int = 4,
        n_heads: int = 2,
        sparse_fusion_blocks: list[int] | None = None,
        dynamic_inference: bool = True,
        shallow_blocks: int = 2,
        lite_blocks: int = 3,
        threshold_easy: float = 0.35,
        threshold_hard: float = 0.65,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dynamic_inference = dynamic_inference
        self.shallow_blocks = shallow_blocks
        self.lite_blocks = lite_blocks
        self.full_blocks = n_blocks

        self.mix_encoder = nn.Linear(n_fft // 2 + 1, emb_dim)
        self.enroll_encoder = nn.Linear(n_fft // 2 + 1, emb_dim)
        self.condition = USEFConditionExtractor(dim=emb_dim, n_heads=n_heads)
        self.separator = TFGridNetLite(
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            n_heads=n_heads,
            sparse_fusion_blocks=sparse_fusion_blocks or [0, 2],
        )
        self.router = DynamicRouter(threshold_easy, threshold_hard)

    def _stft_mag(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        window = torch.hann_window(self.n_fft, device=wav.device, dtype=wav.dtype)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )
        return spec, spec.abs()

    def forward(self, mixture: torch.Tensor, enrollment: torch.Tensor) -> CafeTSEOutput:
        """Forward pass.

        Args:
            mixture: [B, samples]
            enrollment: [B, samples]
        Returns:
            CafeTSEOutput.wav: [B, samples]
        """
        start = time.perf_counter() if not self.training else None

        mix_spec, mix_mag = self._stft_mag(mixture)
        _, enroll_mag = self._stft_mag(enrollment)

        # [B, F, T] -> [B, T, F] -> linear -> [B, T, C]
        mix_tokens = self.mix_encoder(mix_mag.transpose(1, 2))
        enroll_tokens = self.enroll_encoder(enroll_mag.transpose(1, 2))
        cond = self.condition(mix_tokens, enroll_tokens)

        complexity = compute_complexity_score(
            mixture,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        if self.dynamic_inference and not self.training:
            routes = self.router(complexity)
            active_blocks = self.router.active_blocks(
                routes,
                shallow_blocks=self.shallow_blocks,
                lite_blocks=self.lite_blocks,
                full_blocks=self.full_blocks,
            )
        else:
            routes = ["full"] * mixture.shape[0]
            active_blocks = [self.full_blocks] * mixture.shape[0]

        est_spec = self.separator(mix_spec, mix_tokens, cond, active_blocks=active_blocks)

        window = torch.hann_window(self.n_fft, device=mixture.device, dtype=mixture.dtype)
        wav = torch.istft(
            est_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            length=mixture.shape[-1],
        )

        rtf = None
        if start is not None:
            elapsed = time.perf_counter() - start
            audio_dur = mixture.shape[-1] / float(self.sample_rate)
            rtf = elapsed / max(audio_dur, 1e-8)

        return CafeTSEOutput(
            wav=wav,
            route=routes,
            complexity_score=complexity,
            active_blocks=active_blocks,
            rtf=rtf,
        )
```

### 14.6 SI-SDR loss

文件：`src/cafe_tse/losses/sisdr.py`

```python
from __future__ import annotations

import torch


def si_sdr(est: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    est = est - est.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    proj = (est * target).sum(dim=-1, keepdim=True) * target / target.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
    noise = est - proj
    ratio = proj.pow(2).sum(dim=-1) / noise.pow(2).sum(dim=-1).clamp_min(eps)
    return 10 * torch.log10(ratio.clamp_min(eps))


def si_sdr_loss(est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -si_sdr(est, target).mean()
```

---

## 15. Curriculum sampler 设计

文件：`src/cafe_tse/datasets/curriculum_sampler.py`

要求：

- 根据 epoch 控制可见 difficulty。
- 支持 easy / medium / hard。
- 支持最终 mixed fine-tune。
- 训练 log 中必须记录每个 epoch 使用了哪些 difficulty，以及样本数。

核心逻辑：

```python
class CurriculumSchedule:
    def __init__(self, stages):
        self.stages = stages

    def allowed_difficulties(self, epoch: int) -> list[str]:
        for stage in self.stages:
            if epoch <= stage["until_epoch"]:
                return stage["difficulties"]
        return self.stages[-1]["difficulties"]
```

训练器中：

```python
allowed = schedule.allowed_difficulties(epoch)
train_dataset.set_allowed_difficulties(allowed)
```

---

## 16. TF-GridNet-Lite 实现要求

文件：`src/cafe_tse/models/tfgridnet_lite.py`

不要求 100% 复刻 ESPnet TFGridNet，但必须具备以下结构特征：

1. 输入为 complex STFT：`[B, F, T]` complex。
2. 转为 token/grid 表征：`[B, T, F, C]`。
3. 每个 block 至少包含：
   - frequency mixing / full-band mixing
   - temporal mixing
   - cross-frame attention 或轻量 self-attention
   - residual connection
4. 支持 `active_blocks`，用于 dynamic inference。
5. 输出 complex STFT estimate，shape 与输入一致。

建议实现方式：

```text
Complex STFT → concat(real, imag) → linear projection
  ↓
N 个 GridLiteBlock
  ↓
linear projection → real/imag residual mask
  ↓
est_spec = input_spec * complex_mask 或 input_spec + complex_residual
```

不要只写普通 Conv1D separator，否则和 TF-GridNet-Lite 命名不符。

---

## 17. 指标实现要求

### 17.1 必须实现

- SI-SDR
- SI-SDRi
- SDR/SIR/SAR，如果依赖可用
- Params
- RTF
- Skip Ratio
- Average Active Blocks

### 17.2 可选实现

- STOI
- PESQ
- speaker similarity

`results/summary_main_table.md` 至少包含：

| Method | SI-SDRi ↑ | SDR ↑ | SIR ↑ | SAR ↑ | Params ↓ | RTF ↓ | Skip Ratio ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mixture | | | | | 0 | 0 | - |
| USEF-TFGridNet | | | | | | | 0 |
| USEF-TFGridNet-Lite | | | | | | | 0 |
| CAFE-TSE | | | | | | | |

---

## 18. 实验计划

### 18.1 主实验

```text
Exp01: USEF-style TFGridNet baseline
Exp02: USEF-style TFGridNet-Lite
Exp03: TFGridNet-Lite + Curriculum Learning
Exp04: CAFE-TSE = Lite + Curriculum + Sparse Fusion + Dynamic Inference
```

### 18.2 消融实验

```text
Ablation 1: CAFE-TSE w/o Curriculum
Ablation 2: CAFE-TSE w/o Dynamic Inference
Ablation 3: CAFE-TSE w/o Sparse Fusion
Ablation 4: CAFE-TSE w/o Lite backbone, if resources allow
```

### 18.3 分难度实验

在 test set 按 difficulty 分组：

```text
easy
medium
hard
```

分别报告：

```text
SI-SDRi / SDR / SIR / SAR / RTF / Skip Ratio
```

预期：

- easy：RTF 下降明显，SI-SDRi 基本不降。
- medium：RTF 有下降，质量小幅下降或接近 baseline。
- hard：多数样本走 full path，质量保持。

### 18.4 Enrollment 鲁棒性实验

控制：

```text
enrollment = 1s / 3s / 5s
enrollment = clean / noisy
```

目标：说明目标线索质量如何影响特征绑定和选择性注意。

---

## 19. Codex 分步构建计划

请 Codex 按以下顺序构建，不要跳步：

### Step 1：项目骨架与配置

- 创建目录结构。
- 写 `requirements.txt`、`pyproject.toml`。
- 写 config loader。
- 写 logger、seed、audio_io。

验收：

```bash
python -m cafe_tse.cli.train --help
python -m cafe_tse.cli.evaluate --help
```

### Step 2：数据与 manifest

- 实现 toy data。
- 实现 LibriMix manifest parser。
- 实现 enrollment 构造。
- 实现 complexity manifest。

验收：

```bash
python -m cafe_tse.cli.prepare_toy_data --out_dir data/toy --num_samples 8
python -m cafe_tse.cli.compute_complexity_manifest --manifest data/toy/toy_manifest.csv --out_manifest data/toy/toy_manifest_complexity.csv
```

### Step 3：模型模块

- 实现 STFT frontend。
- 实现 USEFConditionExtractor。
- 实现 SparseConditionFusion。
- 实现 TFGridNetLite。
- 实现 DynamicRouter。
- 实现 CafeTSE wrapper。

验收：

```bash
pytest tests/test_model_forward.py -q
```

### Step 4：训练与评估

- 实现 TSEDataset。
- 实现 collate。
- 实现 SI-SDR loss。
- 实现 trainer/evaluator。
- 实现 checkpoint 保存和 resume。

验收：

```bash
bash scripts/harness_smoke.sh
```

### Step 5：实验脚本

- 实现主实验 configs。
- 实现 ablation configs。
- 实现 summarize_results。

验收：

```bash
bash scripts/harness_toy_train_eval.sh
```

### Step 6：LibriMix small 实验

- 使用 100-2000 条 subset 先跑。
- 评估 main table 和 ablation table。

验收：

```bash
bash scripts/harness_librimix_small.sh
```

---

## 20. Git 忽略规则

`.gitignore` 必须包含：

```gitignore
data/raw/
data/processed/
data/downloads/
third_party/
experiments/*
results/*
*.pt
*.pth
*.ckpt
*.wav
__pycache__/
*.pyc
.env
.DS_Store
```

保留：

```text
experiments/.gitkeep
results/.gitkeep
third_party/.gitkeep
```

---

## 21. 最终交付物

完成代码后，应有：

```text
1. 可运行代码
2. README.md
3. configs/*.yaml
4. scripts/harness_*.sh
5. results/summary_main_table.md
6. results/summary_ablation_table.md
7. results/audio_samples/
8. 训练日志和 loss 曲线
9. 模型结构图或 pipeline 图
10. 可写进课程报告的实验分析
```

---

## 22. 报告中可直接使用的方法描述

> 本项目提出 CAFE-TSE，一种基于课程学习与频域复杂度感知的高效目标说话人提取系统。模型以 USEF-style 目标说话人条件建模机制和 TF-GridNet-Lite 时频域分离主干为基础，通过目标说话人 enrollment 语音提供自上而下的注意线索，并利用 cross-attention 完成目标特征绑定。训练阶段，系统根据说话人数、SIR、SNR、重叠比例和 enrollment 质量构造由易到难的 curriculum learning 过程，使模型逐步适应复杂鸡尾酒会场景。推理阶段，系统根据 VAD 比例、频谱熵和能量变化估计输入片段复杂度，并动态选择浅层或完整分离路径，从而在尽量保持目标说话人提取效果的同时降低平均推理开销。

---

## 23. 一句话提醒

本项目最重要的不是把模型做得最大，而是形成完整可验证的系统：

```text
目标说话人提取 + 课程学习 + 动态推理 + 效率指标 + 消融实验
```

只要这条主线跑通，就能很好对应课程要求。

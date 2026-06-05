# 模型根因定位记录

## 结论

当前 1 dB 左右的结果不是课程目标误会，也不是数据/损失天然无法达到 5 dB。使用开源 SpeakerBeam 目标说话人提取模型，在同一份 balanced10 manifest、同一套 SI-SDR 训练目标和同一评估流程上，30 epoch 后达到：

| 模型 | Valid SI-SDR | Test SI-SDR | Test SI-SDRi | Test SDR | Test SIR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 自写 CAFE-TSE teacher `mini_exp44` | 1.59 | 1.06 | 1.27 | 1.06 | 4.56 |
| 开源 SpeakerBeam `open_speakerbeam_balanced10_mid` | 7.49 | 7.00 | 7.21 | 6.99 | 15.89 |
| 开源 SpeakerBeam `open_speakerbeam_balanced40_mid` | 5.52 | 5.45 | 5.65 | 5.44 | 11.07 |
| 开源 SpeakerBeam `open_speakerbeam_shared_clean80_mid` | 10.38 | 10.32 | 10.32 | 10.34 | 22.23 |

因此根因在自写模型实现/结构，尤其是目标说话人条件建模和分离器主体，不在 teacher/student 训练流程本身。

## 开源模型锚点

使用仓库：

- `third_party/speakerbeam`
- 来源：`https://github.com/BUTSpeechFIT/speakerbeam`
- 模型类：`src/models/td_speakerbeam.py::TimeDomainSpeakerBeam`
- 说明：该实现是 TD-SpeakerBeam target speech extraction，并基于 Asteroid Conv-TasNet。

为避免污染现有 conda 环境，Asteroid 代码以 `--target third_party/asteroid_site --no-deps` 安装，训练时使用：

```bash
env LD_LIBRARY_PATH=/root/miniconda3/envs/cafe-tse/lib \
  PYTHONPATH=src:third_party/asteroid_site:third_party/speakerbeam/src \
  /root/miniconda3/envs/cafe-tse/bin/python scripts/train_open_speakerbeam.py ...
```

训练脚本：

- `scripts/train_open_speakerbeam.py`

该脚本只负责读取本项目 manifest、调用开源 `TimeDomainSpeakerBeam`、用 SI-SDR loss 训练和评估；模型实现没有手写替代。

## 关键诊断

### 1. 数据和 loss 是可达的

同一份 `data/metadata/librispeech_tse_balanced10/*_manifest_final.csv` 上，开源 SpeakerBeam 能稳定超过 5 dB。进一步使用 40 个说话人的 `data/metadata/librispeech_tse_balanced40/*_manifest_final.csv`，2400 条训练样本和 400 条测试样本，也能达到 Test SI-SDR 5.45 dB。最终使用 clean shared-speaker 80-speaker 数据 `data/metadata/librispeech_tse_shared_clean80/*_manifest_final.csv`，8000 条训练样本和 800 条测试样本，达到 Test SI-SDR 10.32 dB。说明 mixture/target 对齐、SI-SDR loss、STFT 或 BSS 指标不是 1 dB 的根因。

### 2. Teacher 本身不正常

原 teacher `mini_exp44_balanced10_base_magmask`：

- train SI-SDR：3.84 dB
- valid SI-SDR：1.59 dB
- test SI-SDR：1.06 dB

它不是“训练正常但学生没学好”，而是 teacher 自己泛化失败。

### 3. 开源模型确实使用 enrollment

在 open SpeakerBeam best checkpoint 上替换 enrollment：

| Enrollment 输入 | Test 子集 SI-SDR |
| --- | ---: |
| 正确 enrollment | 7.36 |
| target 自身当 enrollment | 7.47 |
| 错误 enrollment | -5.73 |
| interferer 当 enrollment | -15.29 |
| zero enrollment | -3.71 |

这说明开源模型不是做普通语音增强，而是真的根据 enrollment 选择目标说话人。

### 4. 原模型拿到强提示也上不去

原 teacher 同样替换 enrollment：

| Enrollment 输入 | Test 子集 SI-SDR |
| --- | ---: |
| 正确 enrollment | 0.99 |
| target 自身当 enrollment | 1.16 |
| 错误 enrollment | -1.36 |
| interferer 当 enrollment | -1.82 |
| zero enrollment | -2.62 |

即使把当前 target 自身作为 enrollment，也没有明显超过 1 dB，说明瓶颈不是 enrollment 文件质量，而是自写模型无法把说话人条件转化为有效 mask。

## 原模型问题位置

### 条件表达太弱

`src/cafe_tse/models/usef_condition.py` 只输出一个全局条件向量：

```python
attended, _ = self.attn(query=mix_tokens, key=enroll_tokens, value=enroll_tokens)
attended = self.norm(attended)
return attended.mean(dim=1)
```

该向量再通过 `SparseConditionFusion` 以 FiLM 方式广播到所有时间/频率位置。这会丢掉 enrollment 中最有用的频率轮廓和时序线索。

### 分离器主体不是成熟 TFGridNet

`src/cafe_tse/models/tfgridnet_lite.py` 是轻量自写结构，和真实 TF-GridNet / Conv-TasNet / SpeakerBeam 相比缺少成熟的高容量 TCN/dual-path 建模、深层 mask 估计和经过验证的条件适配层。它能单样本 overfit 到 25 dB，但在真实 split 上只学到弱泛化规则。

### 条件注入不够强

`src/cafe_tse/models/sparse_fusion.py` 中 FiLM 条件是全局通道调制：

```python
delta = gamma * x_norm + beta
return x + delta
```

相比 SpeakerBeam 在 TCN 中间层使用 enrollment embedding 做目标说话人适配，这种全局广播调制太粗，不能稳定地产生目标说话人 mask。

## 后续建议

1. 主结果先以开源 SpeakerBeam 作为正确 baseline，保证报告中有达标模型。
2. 自写 CAFE-TSE 改进应以 SpeakerBeam 为 teacher 或替换 separator 主干，而不是继续微调现有 `TFGridNetLite`。
3. 若保留自写创新点，应把它定位为轻量/动态推理消融，并明确当前版本是失败案例；不要把它作为最终高分模型。
4. 数据生成继续使用修复后的 `prepare_librispeech_tse.py`，保证 enrollment 来自对应 split pool，并在 manifest 中记录 source path。

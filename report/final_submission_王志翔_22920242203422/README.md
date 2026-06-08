# TD-SpeakerBeam 目标说话人提取 —— 课程项目

## 概述

本文件夹为《认知与计算》课程最终提交包，报告涵盖三项任务：

1. **任务一**：人工智能与人类智能的多维度对比分析
2. **任务二**：鸡尾酒会问题模拟：基于 TD-SpeakerBeam 的目标说话人提取系统
   - Shared-clean80 训练与评估（强教师 12.52 dB，中等学生 10.43 dB，fine-tune 蒸馏 10.59 dB，多参考聚合 10.80 dB）
   - 参考语音 sanity check（正确/打乱/干扰/零/短/带噪参考）
   - 混合 SNR 与三说话人压力测试
   - EGSP 频谱预加重诊断
   - **第 2.9 节**：附加挑战实验（流式处理、DEMAND 真实噪声、中英跨语言测试、交互式 Web 展示）
3. **任务三**：神经网络与反向传播的人类学习机制映射

## 附加挑战实验（第 2.9 节）

| 实验 | 脚本 | 关键结果 |
|------|------|----------|
| 流式处理  | `src/scripts/streaming_demo.py`    | 32 s 音频 61 chunks，稳态 RTF 0.0028（350 倍实时），整体 RTF 0.018（55 倍实时） |
| 真实噪声  | `src/scripts/test_real_noise.py`   | DEMAND 咖啡厅噪声：28 样本，+20/+10/+5 dB，SI-SDR 单调下降（11.86 → 11.18 → 7.64 → 4.12 dB） |
| 跨语言    | `src/scripts/test_cross_lingual_batch.py` | 5 对/方向，zh→en 均值 −30.88 dB，en→zh 均值 −44.37 dB，全部失败 |

数据来源（CC BY 4.0 / Apache 2.0）：`src/results/data_sources.md`

## 交互式 Web 展示

在浏览器中打开 `src/gui/index.html` 可查看：
- 系统总览（架构图与核心指标）
- 3 组可交互音频案例（A/B 对比播放）
- 实验图表（雷达图、柱状图）
- 认知映射（选择性注意到模型组件）

所有 demo 音频位于 `demo_audio/`（来自 shared-clean80 测试集实际输出）。

## 项目结构

```
final_submission/
├── 22920242203422+王志翔.pdf          （已编译课程报告，29 页）
├── README.md                          （本文件）
├── requirements.txt                   （Python 依赖）
├── demo_audio/                        （3 组 demo 案例：mixture、target、baseline、ours）
└── src/
    ├── cafe_tse/                      （CAFE-TSE 原型 + 共享工具库）
    ├── gui/                           （交互式 Web 展示）
    │   ├── index.html
    │   ├── styles.css
    │   ├── app.js
    │   └── assets/
    ├── scripts/                       （训练、评估与附加实验脚本）
    │   ├── streaming_demo.py              （流式逐 chunk 推理）
    │   ├── test_real_noise.py             （真实噪声鲁棒性测试）
    │   ├── test_cross_lingual.py          （跨语言单对测试）
    │   ├── test_cross_lingual_batch.py    （跨语言批量测试）
    │   ├── train_open_speakerbeam.py      （TD-SpeakerBeam 训练）
    │   ├── evaluate_open_speakerbeam_variants.py  （参考语音变体评估）
    │   └── ...                            （更多训练/评估脚本）
    └── results/                       （实验输出与可视化）
        ├── data_sources.md            （数据来源与许可证）
        ├── summary.md                 （实验总结）
        ├── figures/                   （实验图表）
        └── ...                        （指标 CSV、JSON 等）
```

## 模型检查点

实验使用 **mid fine-tune distill** 模型：
`open_speakerbeam_shared_clean80_student_mid_distill_ft_continue_w005/best.pt`

测试集（shared-clean80，800 样本）：SI-SDR 10.59 dB，SI-SDRi 10.59 dB，SDR 10.60 dB。

## 范围说明

本系统通过 enrollment 条件建模选择性注意与目标说话人声纹绑定，重点关注单通道目标驱动分离。系统未显式建模空间声源定位、DOA、双耳线索、麦克风阵列或空间角度标签。

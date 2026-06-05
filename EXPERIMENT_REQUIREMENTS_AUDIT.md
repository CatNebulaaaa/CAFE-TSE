# 鸡尾酒会语音分离实验要求核对

本文档依据 `作业要求/课程报告题目-认知与计算-2026.pdf`、`作业要求/课程报告评分标准-认知与计算-2026.pdf`、`作业要求/课程报告的编写格式规范-认知与计算(1).pdf`、`作业要求/课程报告模板-认知与计算(1).docx` 和 `作业要求/Speech Separation Project（参考指南）.pdf` 核对当前系统状态。

## 课程格式

| 要求 | 当前状态 | 处理 |
| --- | --- | --- |
| 封面使用模板格式 | 原 LaTeX 封面字体和布局与 Word 模板不一致 | 已改为模板式封面，填入姓名、学号、年级、专业方向、任课教师 |
| 正文题目三号黑体、居中 | 原第二页重复出现课程报告和个人信息块 | 已改为论文题目居中，姓名和年级/专业/学号按格式规范列出 |
| 正文五号宋体、节标题四号宋体、小标题五号黑体 | 基本满足 | 保持 ctex `zihao=5` 与现有标题格式 |
| 页码居中 | 已满足 | 封面不编号，正文从 1 开始 |

## 实验与交付要求

| 指南要求 | 当前状态 | 结论 |
| --- | --- | --- |
| 方案 A 盲分离或方案 B 定向提取 | 采用方案 B：mixture + enrollment -> target speech | 已满足 |
| 输入多人混合语音，输出目标说话人音频 | MiniLibriMix 2-speaker mixture，输出 target speaker | 已满足；2-speaker 属于指南的 2-3 人范围 |
| 至少 3 位不同说话者 | MiniLibriMix validation/test source metadata 中远超 3 位 | 已满足 |
| 每人不少于 20 条纯净语音 | 完整 MiniLibriMix val metadata 中 `mix_clean` 有 31 位 source speaker 达到 20 条，`mix_both` 有 34 位达到 20 条 | 源数据满足；下一步构造受控 speaker-balanced 子集用于更强证明 |
| 至少 50 段混合测试样本 | test split 100 mixtures | 已满足 |
| 引入餐厅/聚会等真实噪声加分项 | 已做 babble-noise 5 dB 和 10 dB mixture robustness | 已满足 |
| STFT/语谱图、分离核心、iSTFT 重建 | CAFE-TSE 使用 STFT、TF-GridNet-Lite、iSTFT | 已满足 |
| SDR/PESQ 等评估 | 已计算 SI-SDR/SI-SDRi/SDR/SIR/SAR/PESQ/RTF/Params/MACs；泄漏修复后的 disjoint 复训练结果为 5-block student SI-SDRi=0.0710、SDR=-0.0352 | 已满足评估流程；STOI 不稳定，不作为主指标 |
| 报告不少于 8 页 | 当前 PDF 11 页 | 已满足 |
| 包含架构图、指标对比、失败案例反思 | 已有架构图、主结果/效率/消融图；已补明确失败案例小节 | 已满足 |
| 至少 3 组直观对比音频 | `demo_audio/` 中有 3 组 mixture/target/baseline/ours | 已满足 |
| 完整源码与 README | `src/`、`scripts/`、`run_*.sh`、`README.md` 均存在 | 已满足 |

## 已完成补强实验

1. 定位 SDR 偏低原因：原残差输出头只能在 mixture spectrum 上做小幅修饰，已改为 `mag_mask`/`complex_mask` 可选输出头，并修正 BSS 评估中 interferer 的归一化。
2. 验证数据和指标上限：oracle IRM 可达到 13.83 dB，说明数据和指标本身存在可达上限。
3. 修复 disjoint enrollment 构造：扫描 `s1/s2` 源目录，并按源目录位置解析 speaker id，避免把干扰说话人错标为目标 speaker。
4. 修复模型实现：补充 TF-GridNet-Lite 的跨频率卷积路径，并去掉 condition extractor 中会稀释 speaker condition 的 mixture residual。
5. 完成无泄漏复训练：`mini_exp42_base_magmask_disjoint_tfgridfix_lr1e3` 得到 SI-SDRi=-0.0048；`mini_exp43_student_magmask_disjoint_tfgridfix_lr1e3` 得到 SI-SDRi=0.0710。
6. 完成工程展示：`gui/index.html` 已实现交互式系统展示台，包含系统总览、架构图、真实案例音频、实验图表和认知机制映射。

## 当前最优结论

最终可信结论不再采用 `reference_anchor_weight` 作为主结果。早期 `mini_exp30_ours_egsp_magmask_finetune` 加锚点超过 5 dB 的结果在 disjoint enrollment 复核中不可复现；最新报告应采用无泄漏复训练结果，并把锚点高分作为数据泄漏排查中的 sanity/failure case。

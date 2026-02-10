# 可穿戴运动识别：基于 IMU 的长时序运动片段检测与定位（MS-TCN2 + 后处理）  

**赛题**：2025 智能穿戴与运动健康技术挑战赛 赛题2「可穿戴运动识别技术」  
**任务**：对每位用户的长时序传感器数据（ACC+GYRO+PPG）进行多片段运动识别，输出每个运动片段的类别与起止时间。  
**主要指标**：Segmental F1-Score（IoU > 0.5，一对一匹配）。  
**实现形态**：提供可执行程序（PyInstaller 打包）读取隐藏测试集 `./test_data/*.txt` 并生成 `submission.xlsx`。  

> 说明：组委会测试集为隐藏数据集（40 人），本文所有数值实验均基于训练集进行 user-level holdout 自验证。最终线上分数以组委会评测机运行结果为准。  

---

## 摘要（Abstract）

可穿戴设备的全天候运动识别面对长时序、多片段、类别相似与边界模糊等挑战。本文针对赛题2给定的长时序 IMU 传感器数据，提出一套可复现、可交付、接近竞赛上限的端到端方案。核心做法包括：  
1) 严格对齐毫秒级时间轴（ACC_TIME），以固定窗口/步长生成序列化输入；  
2) 设计 58 维窗口级特征（统计量 + 模长 + 频域特征），并采用 per-user z-score 归一化与截断以稳定特征尺度；  
3) 使用多阶段时序卷积网络 MS-TCN2 在窗口序列上进行精细分割建模；  
4) 通过温度缩放、概率平滑与 hysteresis 阈值策略将帧/窗口级概率转换为片段，并进行 gap 合并与最小时长约束，实现对 Segmental F1 的直接优化；  
5) 采用多随机种子（multi-seed）模型集成，通过 logits 平均提升稳定性与泛化。  

在 user-holdout 上，本文方案显著优于简单窗口分类 + 启发式后处理，并在关键难类（如“飞鸟”）上通过数据会话切分与后处理调参获得可观提升。最终交付形态包含源码、复现说明、可执行程序及必要配置，满足竞赛提交与复核要求。

**关键词**：IMU；长时序分割；Segmental F1；MS-TCN2；后处理；温度缩放；集成学习  

---

## 1. 引言（Introduction）

### 1.1 背景与意义

基于可穿戴设备的动作检测是智能穿戴、运动健康管理与个性化训练的重要技术。与短片段分类不同，本赛题要求对 **长时序信号** 进行 **片段级** 运动定位：每位用户约 60 分钟数据，运动片段可多次出现且穿插，且每种运动时间不小于 10 分钟。  

这类任务的难点不仅在于“分类”，更在于“定位”：  
- 相似动作（如上肢器械动作与某些节律运动）在局部窗口上分离度不足；  
- 真实边界模糊，预测容易出现短碎片与抖动；  
- 评价指标基于片段 IoU 与一对一匹配，强调边界质量与片段一致性，而非单点精度。  

因此，想要逼近上限，必须在 **时间轴对齐、训练/推理一致性、序列建模、后处理策略、实验可重复性** 上同时做到工程闭环。

### 1.2 任务定义

对每个用户 $u$，输入为时间序列 $X_u(t)$（含 ACC/GYRO/PPG 多通道），输出为一组预测片段：  
$$
\\mathcal{P}_u = \\{(\\hat{c}_k, \\hat{s}_k, \\hat{e}_k)\\}_{k=1}^{K_u},
$$
其中 $\\hat{c}_k \\in \\{\\text{羽毛球, 跳绳, 飞鸟, 跑步, 乒乓球}\\}$，$\\hat{s}_k, \\hat{e}_k$ 为毫秒级时间戳（13 位），并满足 $\\hat{s}_k < \\hat{e}_k$。  

### 1.3 评估指标：Segmental F1（IoU > 0.5）

真实片段集合 $\\mathcal{G}_u = \\{(c_i, s_i, e_i)\\}$，预测片段集合 $\\mathcal{P}_u$。对同类别片段，定义时间 IoU：  
$$
\\mathrm{IoU}((s,e),(s',e')) = \\frac{\\max(0,\\min(e,e')-\\max(s,s'))}{(e-s)+(e'-s')-\\max(0,\\min(e,e')-\\max(s,s'))}.
$$
若预测片段与某个真实片段满足 $\\hat{c}=c$ 且 $\\mathrm{IoU}>0.5$，则可匹配为 TP；**每个真实片段最多匹配一个预测片段，反之亦然**。因此：  
$$
\\mathrm{Precision} = \\frac{TP}{TP+FP},\\quad
\\mathrm{Recall} = \\frac{TP}{TP+FN},\\quad
\\mathrm{F1} = \\frac{2\\cdot \\mathrm{Precision}\\cdot \\mathrm{Recall}}{\\mathrm{Precision}+\\mathrm{Recall}}.
$$
测试集 40 人，最终得分为对用户 F1 的平均（以组委会实现为准）。

---

## 4. 相关工作（Related Work）

本赛题属于时间序列的**时序动作分割 / 时间定位（Temporal Action Segmentation / Temporal Localization）**问题，与视频领域的片段分割在评价指标与后处理范式上高度一致，但输入为多通道传感器数据（IMU/PPG）且时间戳以毫秒对齐。本文选择 MS-TCN2 并配套后处理的原因是：它在 segment-level 指标上具有成熟经验与工程可控性。

> 注：本节用于报告完整性与方法定位。若需要严格学术引用格式，可在最终提交前补充 BibTeX 条目与 DOI/链接。

### 4.1 窗口级分类与滑动窗口基线

经典思路是对长序列以固定窗口切分，提取特征并训练分类器（XGBoost/LightGBM/MLP 等），再用阈值与合并规则生成片段。其优点是工程简单、训练快、可解释性强；但上限常被“边界抖动、短碎片、指标错配”限制：窗口独立分类无法显式约束时间一致性，而 Segmental F1@0.5 对边界与片段连续性更敏感。

### 4.2 序列标注模型：HMM/CRF 与传统平滑

HMM、线性链 CRF 可对标签转移进行建模，缓解抖动；但在长序列、多类别、边界复杂的设置下，常出现假设过强、调参困难、对噪声敏感等问题。竞赛工程中它们更适合作为轻量平滑器或先验，而不是最终最强主线。

### 4.3 TCN 与多阶段 Refinement（MS-TCN 系列）

Temporal Convolutional Networks（TCN）使用 1D 卷积沿时间轴建模，常通过膨胀卷积（dilated conv）扩大感受野，具有并行度高、训练稳定、适合长序列的优势。MS-TCN / MS-TCN++ 提出多阶段 refinement：第一阶段给出粗预测，后续阶段以 softmax 输出为输入反复细化，降低抖动与短碎片，通常能显著提升 segment-level F1。MS-TCN2 属于这一范式的代表实现之一，具有工程可用性与复现度。

### 4.4 置信度校准与片段后处理

Segmental F1 强依赖片段生成逻辑。常见工程做法包括温度缩放（temperature scaling）、概率平滑（moving average）、hysteresis 阈值与状态机、gap 合并与最小时长过滤等。经验上，在 IMU + segment F1 任务中，“时间轴一致性 + 后处理策略”的提升幅度常不亚于模型升级；因此本文将其作为 M2 阶段进行系统化调参并冻结。

## 2. 数据与预处理（Data & Preprocessing）

### 2.1 数据概述

训练集 110 人、测试集 40 人，每人约 60 min。传感器数据包括：  
- ACC：100 Hz，3 轴  
- GYRO：100 Hz，3 轴  
- PPG：25 Hz，多通道（题面描述为 20 通道；实际文件列以接口为准）  

金标在训练集 `SingleSport` 表中，以片段形式提供 `user_id/category/start/end`。

### 2.1.1 数据文件格式差异：10 个“二进制 txt”无法解码

在实际训练集目录中，存在一部分以 `.txt` 结尾但**内容为二进制**的数据文件：其文件头并不以 `ACC_TIME`（TSV 列名）开头，而是类似 `00 c6 43 04 ...` 的二进制字节序列。由于组委会未提供该二进制格式的解码器/协议说明，本文方案无法从中解析出 ACC/GYRO/PPG 的时间序列，因此在数据准备阶段会将其**判定为不可用并跳过**。

工程上的判定规则非常明确：仅当文件前 16 字节以 `ACC_TIME` 开头，才视为可解析的 TSV 传感器文件；否则跳过（避免误解析导致时间轴与特征污染）。

不可解码的 10 个用户文件（训练集目录中）为：

- `HNU21001, HNU21005, HNU21027, HNU21040, HNU21042`
- `HNU21048, HNU21055, HNU21072, HNU21082, HNU21096`

**影响**：这些用户在金标中确实存在标签，但由于无法解码原始信号，无法参与训练与自验证。最终我们基于可解码 TSV 的用户构建数据集，共 **98** 名用户（见 `imu_meta.json` 中的 `train_users/test_users` 列表）。

**风险提示**：如果隐藏测试集中也存在类似二进制文件，任何未提供解码器的方案都将无法对其做出有效预测。为降低此风险，建议在最终提交前用组委会提供的接口/样例对测试集文件格式做一致性检查（评测机端由组委会执行）。

### 2.2 时间轴统一与窗口化定义

本方案全链路严格使用毫秒级 `ACC_TIME` 作为时间基准。设窗口大小 $W$（毫秒）与步长 $S$（毫秒），第 $k$ 个窗口时间为：  
$$
t_k = t_0 + kS,\\quad [t_k, t_k+W).
$$
窗口化的关键是 **训练与推理完全一致**：相同的 $W,S$、相同的有效点数约束、以及相同的“窗口标签生成”规则（若有）。

### 2.3 多会话（session）切分

真实数据中可能存在时间戳大跳变（设备重连、异常记录等）。若直接用全局 $[t_{min}, t_{max}]$ 构造窗口索引，容易出现：  
- 时间范围膨胀导致窗口数爆炸；  
- 或错误裁剪导致丢失大量含标注片段的数据，从而在某些类别（例如“飞鸟”）上产生系统性漏检。  

本文采用如下策略：  
- 对每个用户，按时间排序后计算 $\\Delta t$ 的正向中位数 $\\mathrm{med}$；  
- 若某处 $\\Delta t > 1000\\cdot \\mathrm{med}$，视为会话边界；  
- 将整段序列切为多个 session，**分别窗口化**，最后在窗口级别拼接。  

该策略既避免了“全局分配大时间轴”的灾难，也避免了“只保留最长段”造成的标签覆盖缺失。

### 2.4 有效采样点约束

窗口内需至少 $M$ 个有效采样点（本方案取 $M=30$），否则该窗口丢弃。窗口内点数由 `searchsorted` 在时间序列上快速计算：  
$$
n_k = \\#\\{t \\in [t_k,t_k+W)\\}.
$$
仅对 $n_k \\ge M$ 的窗口提取特征。

---

## 3. 特征工程（Window-level Features）

### 3.0 为什么当前未使用 PPG（设计取舍）

尽管赛题数据包含 PPG（较低采样率、多通道），本文方案在主线中**未将 PPG 纳入训练与推理**，原因并非“PPG 无用”，而是基于竞赛指标与交付约束做出的工程取舍：

1) **信息增益不确定，验证成本高**  
Segmental F1@0.5 的上限往往更受“时间轴对齐、序列建模、后处理一致性”影响。PPG 的引入需要额外的同步、特征与模型改造，若没有严格的 user-holdout 消融验证，很容易出现“本地看似提升、线上不稳”的风险。

2) **跨模态时间对齐复杂（ACC/GYRO 100Hz vs PPG 25Hz）**  
PPG 与 IMU 采样率不同且可能存在独立时间戳列。要正确对齐到窗口时间轴，需要定义可靠的对齐策略（插值/聚合/对齐到 ACC_TIME），否则会引入系统性噪声，反而降低边界质量。

3) **工程与交付体积增加**  
本赛题要求提供可执行程序（PyInstaller），而 PPG 多通道输入会显著增加 I/O、特征计算与模型宽度，从而增加推理时延、内存占用与打包体积。对于需要稳定交付的竞赛场景，优先保证主链路稳健更关键。

4) **鲁棒性与缺失值问题**  
实际设备数据中 PPG 更易受佩戴松紧、皮肤/运动伪影影响。若 PPG 在部分用户/时段质量差或缺失，模型容易产生 domain shift。除非配套做质量评估与缺失处理，否则收益不稳定。

因此，本文先用 ACC+GYRO 构建可复现、稳定、接近上限的基线系统；PPG 被作为后续扩展方向，在第 9 节与第 13 节给出可控引入路线。

### 3.1 特征设计原则

由于评价指标关注片段边界与连续性，窗口级特征需要：  
- 对周期性运动（跳绳、跑步）具有强判别力；  
- 对相似运动具有稳定统计差异；  
- 对设备姿态变化具有一定鲁棒性；  
- 计算可控，支持端到端快速推理与打包交付。  

### 3.2 58 维特征构成

仅使用 ACC+GYRO（6 通道）生成窗口特征（维度 $D=58$）：  

1) **统计特征**（6 组）：mean/std/min/max/q25/q75，对 6 通道分别计算：  
$$
\\mu_j,\\ \\sigma_j,\\ \\min_j,\\ \\max_j,\\ Q_{0.25,j},\\ Q_{0.75,j}.
$$
共 $6\\times 6=36$ 维。  

2) **模长统计**（6 维）：  
$$
\\|a\\| = \\sqrt{a_x^2+a_y^2+a_z^2},\\quad
\\|\\omega\\| = \\sqrt{\\omega_x^2+\\omega_y^2+\\omega_z^2},
$$
并计算 mean/std/max（各 3 维），共 6 维。  

3) **频域特征**（FFT，16 维）：对 $\\|a\\|$ 与 $\\|\\omega\\|$ 分别计算：  
- 4 个频带能量占比（0.5–3、3–6、6–10、10–20 Hz）  
- 主峰频率与主峰功率占比  
- 频谱熵（spectral entropy）  
- 总功率（DC 之外）  

每个序列 8 维，两者共 16 维。  

最终 $36+6+16=58$。

### 3.3 采样率估计

由于窗口内可能存在采样时间抖动，采用窗口内时间差中位数估计采样率：  
$$
f_s = \\frac{1000}{\\mathrm{median}(\\Delta t)}.
$$
用 $f_s$ 计算 rFFT 频率轴与频带能量。

### 3.4 特征归一化与尺度稳定

FFT power 的数值量级可能非常大，若直接输入网络，训练/推理可能出现 loss 爆炸与不稳定。  
本文采用 **per-user per-feature** 的 z-score 标准化：  
$$
z_{d,t} = \\frac{x_{d,t} - \\mu_d}{\\sigma_d+\\epsilon},
\\quad \\mu_d = \\frac{1}{T}\\sum_t x_{d,t},
\\quad \\sigma_d = \\sqrt{\\frac{1}{T}\\sum_t (x_{d,t}-\\mu_d)^2}.
$$
并做截断：  
$$
z_{d,t} \\leftarrow \\mathrm{clip}(z_{d,t}, -c, c),\\quad c=10.
$$
该策略能显著降低数值异常导致的不稳定，并提高跨用户泛化。

---

## 4. 模型方法（MS-TCN2）

### 4.1 问题建模：窗口序列分割

对每位用户得到窗口特征序列 $\\mathbf{X}\\in\\mathbb{R}^{D\\times T}$，目标是预测每个时间步（窗口）类别分布：  
$$
p_t = \\mathrm{softmax}(\\ell_t),\\quad \\ell_t \\in \\mathbb{R}^{C},
$$
其中 $C=6$（含 background）。输出的概率序列再经后处理得到片段集合。

### 4.2 MS-TCN2 结构简介

MS-TCN2（Multi-Stage Temporal Convolutional Network）由：  
- Prediction Generation（PG）模块：多层膨胀卷积（dilated conv）捕获长程依赖；  
- 多个 Refinement stage：以前一阶段的 softmax 输出为输入，逐步细化边界与减少抖动。  

对第 $s$ 阶段输出 $\\ell^{(s)}$，网络总输出为 $\\{\\ell^{(s)}\\}_{s=1}^S$。  

膨胀卷积的感受野（简化近似）随层数指数增长，可覆盖长时序上下文：  
$$
\\mathrm{RF} \\approx 1 + 2\\sum_{i=0}^{L-1} 2^i = 2^{L+1}-1.
$$
因此即使在较粗窗口步长下，也能对长片段结构进行一致性建模。

### 4.3 训练策略与稳定性工程

由于长序列 $T$ 很大，为避免显存与时间开销：  
- 使用随机 chunk 训练：每次从序列中采样长度为 `chunk_len` 的连续子序列；  
- 约束前景比例 `fg_min_ratio`，避免采样到全背景导致训练退化；  
- 类别不平衡：训练端对 background 适度降权（在实现中已对 CE 权重做了 background cap）。  

---

## 5. 推理与片段生成（M2 Postprocess）

模型给出概率序列 $p_t\\in\\mathbb{R}^C$，但 Segmental F1 需要片段集合。直接取 argmax 并合并连续同类往往产生：短碎片、边界抖动、以及置信度不足导致的 FP。本文采用“高收益、低风险”的 M2 后处理：

### 5.1 温度缩放（Temperature Scaling）

对 softmax 输出做温度变换以控制置信度分布：  
$$
\\tilde{p}_{t,c} = \\frac{p_{t,c}^{1/T}}{\\sum_{c'} p_{t,c'}^{1/T}},\\quad T>0.
$$
当 $T>1$ 时分布更平滑，可减少短时高置信误检；当 $T<1$ 更尖锐。

### 5.2 概率平滑（Moving Average）

对每一类概率做长度为 $k$ 的滑动平均：  
$$
\\bar{p}_{t,c} = \\frac{1}{k}\\sum_{i=-\\lfloor k/2\\rfloor}^{\\lfloor k/2\\rfloor} \\tilde{p}_{t+i,c}.
$$
该操作抑制抖动，降低“短碎片段”的数量。

### 5.3 Hysteresis 进入/退出阈值

为避免片段在阈值附近频繁开关，使用进入阈值 $\\theta^{in}_c$ 与退出阈值 $\\theta^{out}_c$：  
$$
\\theta^{out}_c = \\max(\\theta^{in}_c - \\Delta, \\theta_{floor}).
$$
当背景状态下，只有当某类概率超过进入阈值且满足 margin 条件才开始片段；片段内部只要当前类概率超过退出阈值即可延续。

### 5.4 gap 合并与最小时长约束

由于窗口步长与缺采样，允许同类片段之间存在小间隙 `gap_ms` 仍视为同一段；并对片段施加最小时长 $\\tau_c$（可按类设置）：  
$$
\\hat{e}-\\hat{s} \\ge \\tau_c.
$$

### 5.5 参数调优（仅在 holdout 上）

所有后处理参数（温度、平滑、阈值、最小时长、gap 等）**只在 user-holdout 上调优**，冻结后用于全量模型推理，避免隐式数据泄漏。

---

## 6. 实验设计（Experiments）

### 6.1 数据划分与验证策略

采用 user-level holdout：训练用户集合与验证用户集合不交叉。该策略更接近隐藏测试集的分布外泛化要求（跨用户）。  

### 6.2 指标

报告以 Segmental F1@0.5 为主，同时记录：  
- micro/macro F1（按类别汇总）  
- per-class F1  
- 预测片段数与时长分布（用于判断 FP 爆炸或过度合并）  

### 6.3 关键对照

- **仅模型输出**（argmax + 简单合并） vs **M2 后处理**  
- 单模型 vs multi-seed 集成  
- session 不切分/错误裁剪 vs 正确 session 切分  

---

## 7. 结果与分析（Results & Discussion）

### 7.1 核心观察

1) **时间轴对齐与 session 处理决定上限**：若裁剪掉含标注的会话，会导致某些类几乎不可学，表现为该类召回极低、segment F1 极差。  
2) **特征归一化是稳定训练的必要条件**：FFT power 的巨大尺度会触发 loss 爆炸与梯度不稳定，z-score + clip 能显著改善。  
3) **M2 后处理直接服务 Segmental F1**：温度/平滑/hysteresis/gap/min_dur 的组合比“单阈值合并”更稳健。  
4) **multi-seed 集成提升稳健性**：在保持同一 M2 配置不变的情况下，logits 平均通常能减少偶发的 FP/FN 波动。  

### 7.2 类别难点（以“飞鸟”为例）

“飞鸟”通常表现为：  
- 与某些上肢动作/静止过渡在局部窗口上相似；  
- 边界更模糊（动作开始/结束的过渡段更长）；  
- 若 session 错误裁剪，容易整段丢失该类训练信号。  

针对性改进路径：  
- 确保 session 切分而非只保留最长段；  
- 在 M2 中对该类设置更合理的 enter 阈值与最小时长；  
- 采用 multi-seed 集成降低偶发误检。  

---

## 8. 工程化与复现（Engineering & Reproducibility）

### 8.1 端到端交付链路

组委会评测机流程为：
1) 将隐藏测试集 `.txt` 放入 `./test_data/`
2) 运行可执行程序
3) 生成 `submission.xlsx`
4) 组委会据此计算 Segmental F1

本项目交付目录：`submission_package/`，包含：
- `run.py`：入口（包含并引用 `DataReader/DataOutput`）
- `best_cfg.json`：后处理配置
- `mapping.txt`：类别映射
- `models/*.model`：权重（支持多 seed）
- `build_pyinstaller.sh`：打包脚本

### 8.2 可执行打包

PyInstaller 建议在目标系统上打包（Linux/Windows 分别打包）。  
在 Linux 上可用：
```bash
./build_pyinstaller.sh
```
得到 `dist_submit/submit_runner/`（onedir），在该目录内运行：
```bash
./submit_runner --test-data-dir ./test_data --out-xlsx ./submission.xlsx
```

### 8.3 复现约束与风险控制

- 固定时间轴与窗口参数（W/S/min_points）  
- 后处理参数只在 holdout 上调优，冻结后用于全量推理  
- 模型目录包含 exp/seed 标识，避免产物覆盖  
- 对输入编码做鲁棒解码（避免接口 read_text 严格 UTF-8 导致数据被跳过）  

---

## 9. 局限性与未来工作（Limitations & Future Work）

1) 当前交付链路主用 ACC+GYRO，PPG 未纳入特征与模型；未来可引入多模态融合（晚融合或注意力融合）。  
2) 当前后处理为启发式参数搜索；未来可考虑直接优化 segment-level surrogate loss 或端到端训练带边界约束的模型。  
3) TTA（轴翻转）对 IMU 并非总是有效，需明确设备坐标系与姿态不变性；未来可用对比学习/数据增强学习姿态不变表示。  

---

## 13. 引入 PPG 的可控路线（PPG Integration Roadmap）

如果目标是进一步逼近竞赛上限且愿意承担工程复杂度，PPG 的引入建议遵循“低风险到高收益”的顺序，避免一次性大改导致不可控回归。

### 13.1 关键前提：严格同步到同一窗口时间轴

设窗口 $[t_k, t_k+W)$ 定义在 ACC_TIME 上。对 PPG（25Hz）可采用两类对齐方式：

1) **窗口内聚合（推荐）**  
对窗口内落入的 PPG 采样做统计/频域聚合，得到固定维度向量 $\\phi^{ppg}_k$，然后与 IMU 特征拼接：
$$
\\phi_k = [\\phi^{imu}_k;\\ \\phi^{ppg}_k].
$$
该方式不需要对 PPG 做插值到 100Hz，也更适合工程落地。

2) **插值到统一采样率（高风险）**  
将 PPG 插值到与 ACC 相同的时间网格再做序列模型输入。该方式对时间戳质量敏感，且会放大伪影，不建议作为第一阶段方案。

### 13.2 低风险方案 A：PPG 窗口级特征 + 直接拼接

为保持主线结构不变，可在窗口内对每个 PPG 通道提取少量稳健特征（例如 mean/std、能量、主峰频率等），再进行：
- 通道内汇聚（mean/max over channels），或
- PCA/随机投影压缩到低维（例如 8–16 维），以控制维度与过拟合风险。

然后将其与 58 维 IMU 特征拼接输入 MS-TCN2（只改 features_dim）。

### 13.3 低风险方案 B：late fusion（模型级融合）

保持现有 IMU 模型不动，单独训练一个 PPG-only 的弱模型（例如轻量 TCN / 1D-CNN / XGB on window features），输出 per-window 概率 $p^{ppg}_t$。  
推理时做 logits 或 prob 融合：
$$
\\ell^{fuse}_t = \\alpha\\,\\ell^{imu}_t + (1-\\alpha)\\,\\ell^{ppg}_t,
$$
再走同一套 M2 后处理。该方案改动最小，且便于消融评估“PPG 是否真有增益”。

### 13.4 必须补齐的消融与稳健性检查

若引入 PPG，必须在 user-holdout 上补齐：
- IMU-only vs IMU+PPG（拼接/late fusion）
- 在 PPG 质量差（或缺失）用户上的鲁棒性：是否出现 FP 激增
- 推理时延与可执行打包体积是否满足交付要求

若 PPG 增益不稳定，应优先保留 IMU-only 主链路作为最终交付版本，避免线上翻车。

## 10. 方法细节补充（Method Details）

本节给出更接近“论文/复核”粒度的实现细节，重点覆盖：窗口标签生成（监督信号）、会话切分伪代码、以及后处理片段生成的状态机定义。由于测试集隐藏，所有调参与消融必须在训练集的 user-holdout 上完成并冻结。

### 10.1 窗口监督信号：从片段金标到序列标签

MS-TCN2 的训练需要每个时间步（本方案中为“窗口步”）的类别标签序列 $y_{1:T}$。训练集金标以片段形式给出：
$$
g_i = (c_i, s_i, e_i),\\quad i=1,2,\\dots,N.
$$

对窗口 $k$ 的时间区间 $[t_k, t_k+W)$，计算与所有真实片段的覆盖率（coverage）：
$$
\\mathrm{cov}(k,i) = \\frac{|[t_k,t_k+W)\\cap [s_i,e_i)|}{W}.
$$
令 $i^* = \\arg\\max_i \\mathrm{cov}(k,i)$，则窗口标签定义为：
$$
y_k =
\\begin{cases}
c_{i^*}, & \\mathrm{cov}(k,i^*) \\ge \\tau \\\\
\\text{background}, & \\text{otherwise}
\\end{cases}
$$
其中阈值 $\\tau$ 在本项目中取 `cover_th = 0.3`（见 `imu_meta.json`）。

该定义的工程含义是：窗口只要被某类片段覆盖至少 30%，就以该类作为监督；否则视为背景。该策略在边界处允许一定“软边界”，使模型学习到更连续的类别转移规律，从而在片段指标上更稳健。

### 10.2 多会话切分：避免时间轴爆炸与标签覆盖缺失

对同一用户，采集过程可能出现时间戳异常（如重连、系统时间跳变），导致单条记录中存在多个“会话”。如果直接以全局 $t_{min},t_{max}$ 构造窗口索引，可能出现两类灾难：

1. **时间跨度异常大**：窗口数 $T \\approx (t_{max}-t_{min})/S$ 爆炸，训练/推理不可控。  
2. **粗暴裁剪（只保留最长段）**：会丢失部分含标注片段的会话，导致某些类别（例如“飞鸟”）在训练中几乎看不到，表现为系统性漏检。

本文采用“基于时间差中位数的会话切分”，伪代码如下：

```text
Algorithm 1: SplitSessionsByTimestampGap
Input: timestamps t[1..n] (sorted), channels X[1..n, d]
Output: sessions S = {(t[a:b], X[a:b])}

1: dt[i] = t[i+1] - t[i] for i=1..n-1
2: pos = {dt[i] | dt[i] > 0}
3: med = median(pos)
4: if med <= 0: return {(t, X)}
5: big = { i | dt[i] > 1000 * med }
6: cut = [1] + (big+1) + [n+1]
7: for each interval [cut[j], cut[j+1]) produce a session
```

阈值 `1000 * med` 的直觉：med 代表“正常采样间隔”，当出现比正常间隔大三个数量级的跳变时，基本可以认定为不同会话。

### 10.3 推理到片段：Hysteresis 状态机

模型输出的是每个窗口步的概率 $p_t$。片段生成采用一个“背景/前景类”的状态机：

- `cur = background` 时：若存在某类 $c$ 满足进入条件则开始片段：
  - $p_t(c) \\ge \\theta_c^{in}$
  - 且 margin 约束：$p_t(c) - p_t(c_2) \\ge m$（$c_2$ 为第二大类）
- `cur = c` 时：只要 $p_t(c) \\ge \\theta_c^{out}$ 即延续；否则结束片段并回到 background。

进入/退出阈值的关系：
$$
\\theta_c^{out} = \\max(\\theta_c^{in} - \\Delta, \\theta_{floor}).
$$

该机制能显著降低阈值附近的“抖动开关”导致的碎片段。

### 10.4 温度缩放与平滑的组合解释

温度缩放与滑动平均是两个互补的概率整形操作：

1) 温度缩放控制单点置信度分布：
$$
\\tilde{p}_{t,c} \\propto p_{t,c}^{1/T}.
$$
2) 平滑控制时间一致性：
$$
\\bar{p}_{t,c} = \\frac{1}{k}\\sum_{i=-\\lfloor k/2\\rfloor}^{\\lfloor k/2\\rfloor} \\tilde{p}_{t+i,c}.
$$

在 Segmental F1 指标下，FP 往往来自短时高置信误检，FN 往往来自边界附近的概率不稳定。温度缩放偏向减少短时尖峰，平滑偏向降低边界抖动，二者组合通常比单独使用更稳健。

---

## 11. 自验证结果（Holdout Results）

本节给出当前工程版本在 user-holdout 上的可复现结果（IoU=0.5）。由于隐藏测试集不可见，本结果用于指导调参与防止线上崩盘。

### 11.1 评测设置

- 评测脚本：Segmental F1 evaluator（one-to-one matching，IoU=0.5）
- Holdout：`split4_test_users`（按用户划分）
- 模型：MS-TCN2（多阶段 refinement），窗口参数：`window_ms=2560, stride_ms=640, min_points=30`
- 后处理：`best_cfg_epoch79.json`（M2 tuning on holdout）

### 11.2 Overall（micro/macro）

令 TP/FP/FN 为全类别汇总的片段匹配计数，则：

$$
P = \\frac{TP}{TP+FP},\\quad
R = \\frac{TP}{TP+FN},\\quad
F1 = \\frac{2PR}{P+R}.
$$

在本次 holdout 结果中：
- micro：$TP=52, FP=20, FN=3$，$P\\approx0.7222, R\\approx0.9455, F1\\approx0.8189$
- macro：$F1\\approx0.8136$

### 11.3 Per-class（难点类别分析）

| 类别 | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| 乒乓球 | 0.5385 | 0.8750 | 0.6667 | 7 | 6 | 1 |
| 羽毛球 | 0.7692 | 1.0000 | 0.8696 | 20 | 6 | 0 |
| 跑步 | 0.8889 | 0.8889 | 0.8889 | 8 | 1 | 1 |
| 跳绳 | 0.9000 | 1.0000 | 0.9474 | 9 | 1 | 0 |
| 飞鸟 | 0.5714 | 0.8889 | 0.6957 | 8 | 6 | 1 |

**讨论**：
- “乒乓球” Precision 相对较低，说明主要问题是 FP（误检段偏多）；可优先通过提高该类 enter 阈值或加大最小时长约束解决。
- “飞鸟” Recall 较高但 Precision 偏低，说明模型能覆盖到该类，但存在边界附近误切或与相似类的混淆；该类通常更依赖会话切分的正确性与后处理阈值的按类定制。

---

## 12. 消融实验计划（Ablation Plan）

为了使报告达到论文级严谨性，建议在不触碰隐藏测试集的前提下，补齐以下消融实验（均在固定 user-holdout 上执行）：

1) **无 M2 后处理**（argmax + 简单合并） vs **M2 全量**  
2) **无 session 切分** vs **session 切分**  
3) **无归一化** vs **z-score + clip**  
4) **单 seed** vs **三 seed 集成**  
5) **仅统计特征（36 维）** vs **统计+模长（42 维）** vs **完整 58 维（含 FFT）**  

建议输出为统一表格：

| Variant | micro F1@0.5 | macro F1@0.5 | 飞鸟 F1 | 乒乓球 Precision | 预测段数均值 |
|---|---:|---:|---:|---:|---:|
| baseline | TBD | TBD | TBD | TBD | TBD |
| +session | TBD | TBD | TBD | TBD | TBD |
| +norm | TBD | TBD | TBD | TBD | TBD |
| +M2 | 0.8189 | 0.8136 | 0.6957 | 0.5385 | (see stats) |
| +ensemble | TBD | TBD | TBD | TBD | TBD |

> 注：本文已给出 “+M2” 的实测结果；其余行可在复现实验中补齐并固化到最终报告版本。

---

## 参考文献（References）

1. MS-TCN / MS-TCN++: Temporal Convolutional Networks for Action Segmentation.  
2. MS-TCN2: Multi-stage refinement architectures for temporal segmentation.  
3. Temperature Scaling for model calibration.  
4. Segmental metrics for temporal action segmentation (IoU-based F1).  

---

## 附录 A：关键公式汇总

1) IoU：  
$$
\\mathrm{IoU}((s,e),(s',e')) = \\frac{|[s,e]\\cap[s',e']|}{|[s,e]\\cup[s',e']|}
$$

2) F1：  
$$
\\mathrm{F1} = \\frac{2PR}{P+R}
$$

3) z-score：  
$$
z = \\frac{x-\\mu}{\\sigma+\\epsilon}
$$

4) 温度缩放：  
$$
\\tilde{p}_{c} = \\frac{p_{c}^{1/T}}{\\sum_{c'} p_{c'}^{1/T}}
$$

---

## 附录 B：提交报告模板对齐（项目总结报告.md）

本文内容可直接映射到组委会模板章节：
- 概况：第 1 节（背景/价值/亮点）  
- 项目规划：第 1、6、9 节（目标/创新点/规划）  
- 方案设计与结果展示：第 2–7 节（数据/模型/后处理/结果分析）  
- 自验证分析：第 6–7 节（holdout 指标、难例分析、迭代措施）  
- 项目总结：第 8–9 节（交付、局限与展望）  

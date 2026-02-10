# Architecture (接近 SOTA 的工程实现要点)

## 1. 目标与评测口径

- 主指标：Segmental F1（IoU > 0.5，one-to-one matching）。
- 自验证：按 `user_id` 分组 holdout；同时监控 per-class F1、预测段数与段时长分布。
- 测试集隐藏：最终分数由组委会在评测机运行程序产生的 `submission.xlsx` 计算。

## 2. 数据与时间轴

- 全链路统一毫秒时间戳（ACC_TIME）。
- 以窗口为基本步：
  - `start_ms = t0 + k * stride_ms`
  - `end_ms = start_ms + window_ms`
- 单个用户可能存在多段会话（timestamp 大跳变）：拆分 session 后分别做滑窗，再拼接窗口序列。

## 3. 特征（Window Features）

输入只用 ACC+GYRO（6 通道）：
- 统计特征：mean/std/min/max/q25/q75（6 组 x 6 通道）
- 模长：acc_mag / gyro_mag 的 mean/std/max（6 维）
- 频域特征：acc_mag + gyro_mag 的 FFT 特征（每个 8 维：band energies + peak + entropy + total power）

总维度：58。

稳定性关键：
- FFT 的 power 量级可能非常大，因此推理端/训练端都做 **per-user per-feature z-score + clip**（clip=10）。

## 4. 模型（MS-TCN2）

- Backbone：MS-TCN2（Prediction Generation + 多阶段 Refinement）。
- 训练策略：全量训练（split_98）用于最终模型；split_4 用于 holdout 调参（M2）。
- 为防止长序列 OOM：训练用随机 chunk（chunk_len=1024）并要求最小前景比例（fg_min_ratio=0.05）。

## 5. 推理与后处理（M2）

推理输出为 per-window 概率 `prob(T, C)`，M2 后处理把概率转换为 segments：
- 温度缩放（temp）
- 概率平滑（smooth）
- Hysteresis enter/exit 阈值（默认阈值 + per-class enter）
- gap 合并与 min_dur_ms（支持按类最小时长）

M2 配置文件：`best_cfg.json`。

## 6. 多 seed 集成

若 `models/` 下存在多个 `.model`：
- 对同一输入窗口序列，分别前向推理得到 logits
- 取 logits 平均后 softmax
- 再走同一套 M2 后处理

收益：通常能提升稳健性并降低单次训练随机性。


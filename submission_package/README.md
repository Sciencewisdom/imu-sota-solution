# Submission Package (赛题2 可执行交付)

本目录用于交付组委会“隐藏测试集”评测所需的**可运行程序**与**必要配置/模型**。

## 1. 赛题要求摘要（来自《竞赛题2_运动识别.md》）

- 任务：对每位测试用户的长时信号，输出每个运动片段的 `category + start + end`。
- 运动类别：`羽毛球 / 跳绳 / 飞鸟 / 跑步 / 乒乓球`（另含 `background` 仅用于内部建模）。
- 评分：**Segmental F1-Score**，IoU 阈值固定为 `0.5`，且 **一个真实片段最多匹配一个预测片段**（one-to-one）。
- 测试集：40 人，数据不公开。组委会会在评测机上提供 `./test_data/*.txt`，运行你的程序得到 `submission.xlsx` 后计算分数。
- 交付物：最终 Python 源码、总结报告、以及可执行文件（.exe，官方建议使用 pyinstaller）。

## 2. 目录内容

- `run.py`：入口脚本。读取 `./test_data/*.txt`，输出 `./submission.xlsx`。
- `input.py` / `output.py`：组委会提供的读取/写出接口示例（原样拷贝并在程序中引用）。
- `mapping.txt`：类别映射（训练时的类别顺序）。
- `best_cfg.json`：M2 推理后处理配置（温度/平滑/阈值/按类最小时长等）。
- `models/*.model`：MS-TCN2 模型权重（可放 1 个或多个；多个时会自动做 logits 平均集成）。
- `mstcn2_model_min.py`：推理用的最小 MS-TCN2 网络定义（避免依赖训练仓库）。
- `build_pyinstaller.sh`：在当前环境构建可执行文件（onedir）的脚本。

## 3. 运行方式（脚本版）

1. 将隐藏测试集放到当前目录的 `./test_data/` 下（组委会评测时会自动提供）。
2. 运行：

```bash
python3 run.py
```

输出：
- `./submission.xlsx`

常用参数：
- GPU：`python3 run.py --device cuda`
- CPU：`python3 run.py --device cpu`

## 4. 可执行文件（PyInstaller）

在本机已验证可生成 Linux 可执行目录（onedir）。如果组委会要求 **Windows .exe**：
- 建议在 Windows 上用同版本 Python + PyInstaller 重新打包（PyInstaller 通常不建议跨平台交叉编译）。

本机构建（Linux）：

```bash
./build_pyinstaller.sh
```

产物：
- `./dist_submit/submit_runner/`

运行示例（在产物目录内）：

```bash
./submit_runner --test-data-dir ./test_data --out-xlsx ./submission.xlsx
```

## 5. 输出格式

输出文件 `submission.xlsx` 包含 4 列：
- `user_id`（str）
- `category`（str）
- `start`（int64, 13 位时间戳毫秒）
- `end`（int64, 13 位时间戳毫秒）

## 6. 重要注意事项

- **测试集隐藏**：本地无法得到官方最终分数，只能用训练集的 user-holdout 做自验证。
- **编码兼容**：入口会做更鲁棒的文本解码，避免因非严格 UTF-8 导致某些用户文件被跳过。
- **权重体积**：如果你放入多个 seed 做集成，推理更稳，但包会更大。


# IMU Temporal Segmentation Solution (MS-TCN2 + M2 Postprocess)

This repository contains a near-SOTA engineered pipeline for the competition task:
- Input: long IMU sensor TSV per user (ACC+GYRO)
- Output: segments (category/start/end) in `submission.xlsx`
- Metric: Segmental F1 @ IoU>0.5 (one-to-one matching)

## Contents

- `submission_package/`: runnable package interface (script + pyinstaller build script)
- `src/`: adapter scripts for preparing MS-TCN2 data, caching inference, and postprocess tuning
- `third_party/MS-TCN2/`: MS-TCN2 training/inference code (patched for seed/exp)
- `docs/`: engineering roadmap and notes

## Quick Start (Submission)

See `submission_package/README.md`.

## Training/Reproduction

High-level:
1. Prepare dataset (features/timestamps/groundTruth/splits) for MS-TCN2.
2. Train MS-TCN2 on split4 (holdout) and tune postprocess (M2).
3. Train MS-TCN2 on split98 (full) for final model; optionally train multi-seed and ensemble.

Implementation details are in `docs/ROADMAP.md` and the scripts under `src/`.

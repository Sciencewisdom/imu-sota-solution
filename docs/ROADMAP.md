# IMU Segmental F1 (IoU>0.5) Roadmap (≈95-96% SOTA Execution Standard)

This document is the working checklist for pushing the current IMU temporal segmentation system toward near-SOTA quality, with **Segmental F1 @ IoU>0.5** as the primary metric.

## Goal Definition

1. **Primary metric**: competition **Segmental F1 (IoU > 0.5)**, with **user-grouped holdout** evaluation.
2. **Secondary views** (must monitor):
   - per-class F1
   - predicted segment count distribution
   - predicted segment duration distribution
3. **"≈96% SOTA level"**: not an absolute score; it means engineering factors are near ceiling:
   - alignment is correct end-to-end
   - results are stable and reproducible
   - inference + postprocess are consistent and batchable
   - ensemble/calibration are integrated cleanly

## Stage 0: Baseline + Evaluation Contract (Lock First)

1. **Time axis is fixed** (entire pipeline uses milliseconds):
   - `t -> start_ms = t0 + t*stride_ms`
   - `end_ms = start_ms + window_ms`
2. **Label rules are fixed** and identical across:
   - data preparation
   - training
   - validation
   - inference
   Parameters include: `cover_th`, `min_points`, `window_ms`, `stride_ms`.
3. **Split is fixed**:
   - strictly grouped by `user`
   - all hyperparameter tuning is done on holdout only
   - final full-train artifacts must not leak into tuning
4. **Artifacts** (must exist and be versioned per experiment):
   - `data/imu_sota/imu_meta.json` (all key params + user list)
   - `config.json` per run (full CLI/config + seed)
   - git hash + environment versions

## Stage 1: MS-TCN++ in the “Strong + Stable” Form (Not Just Smoothing)

1. **Input representation upgrade**:
   - from only `prob(T×C)` to:
     - `X_t = concat([window_feats(D), logprob(C)])` → `T×(D+C)`
   - store and track `features_dim` and the exact construction logic.
2. **Training strategy**:
   - long sequences must use **foreground-aware chunk sampling**
   - each batch must contain enough foreground ratio to avoid background collapse.
3. **Class imbalance**:
   - downweight `background` or use focal-like weighting
   - foreground classes get higher weight
   - always monitor per-class recall.
4. **Multi-stage supervision**:
   - compute CE for **every stage**
   - keep temporal smoothing regularizer (adjacent KL/L2)
   - padding mask must fully exclude padding from all losses.
5. **Inference consistency**:
   - run full sequence or overlap sliding
   - use overlap stitching that avoids edge effects (e.g., keep center region).
6. **Acceptance for Stage 1**:
   - on the fixed holdout split, MS-TCN++ (feat+logprob) must beat:
     - “window classifier + heuristic postprocess”
   - variance across seeds should be small enough for reliable comparisons.

## Stage 2: Systematic Tuning + Integration (Core to Reach the Ceiling)

1. **Per-class postprocess parameters** (minimum):
   - `prob_th`, `min_dur`, `gap`, `margin`, `boundary_shift` (per-class, not global)
2. **Calibration**:
   - temperature scaling (per-model or per-class) fitted on holdout
   - goal: reduce short false positives.
3. **Ensemble**:
   1. multi-seed: average logits/probabilities on the same aligned time axis
   2. TTA flip: 8x axis flips, fuse, then postprocess
   3. model fusion: MS-TCN++ + strong window classifier (XGB) via late fusion
      - define fixed merge/NMS rules for conflicts.
4. **Auto-search**:
   - random search / Bayesian optimization on holdout
   - outputs `best_config.json` and is versioned.
5. **Acceptance for Stage 2**:
   - stable improvement across **3 different seeds** of the user-holdout
   - predicted segment count does not explode.

## Stage 3: Leakage Prevention + Engineering (Avoid “Looks Great, Fails Online”)

1. **Data lineage**:
   - cache filenames include:
     - `window/stride/cover_th/feat_hash/model_epoch`
   - forbid mixing caches from different configs.
2. **One-click reproducibility**:
   - full pipeline scripts:
     - `prepare -> train -> infer_cache -> tune_postprocess -> infer_submit`
   - default GPU path; record time + VRAM usage.
3. **Submission pipeline**:
   - start from official `test_data/*.txt`
   - reuse exact same feature + time-axis logic
   - output `submission.xlsx`
   - self-checks:
     - column names
     - time units
     - `start < end`
     - no empty / NaNs.
4. **Acceptance for Stage 3**:
   - end-to-end run succeeds in a clean directory
   - results reproducible.

## Stage 4: Final Train + Submission (Once Only, Avoid Contamination)

1. With Stage 2 fully frozen (all hparams + postprocess), run **full training** (optionally multi-seed).
2. Inference with the frozen fusion strategy to generate final `submission.xlsx`.
3. Outputs:
   - `models/final/*`
   - `reports/final/*`
   - `submission.xlsx`
   - `README_run.md`

## Default Milestones

1. **M1 (today)**:
   - Stage 0 + Stage 1 complete
   - MS-TCN++ (feat+logprob) beats best baseline on holdout.
2. **M2 (1-2 days)**:
   - Stage 2 complete
   - multi-seed + TTA + calibration + per-class postprocess to reach stable high-score band.
3. **M3 (afterwards)**:
   - Stage 3 + Stage 4 complete
   - end-to-end pipeline is submit-ready and reproducible.


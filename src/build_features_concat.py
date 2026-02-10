#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Build concatenated MS-TCN2 features: [window_feats, logprob] -> (D+C,T).")
    ap.add_argument("--data-root", default="/tmp/项目/repo/MS-TCN2/data/imu_sota")
    ap.add_argument("--prob-dir", default="/tmp/项目/repo/MS-TCN2/data/imu_sota/xgb_prob")
    ap.add_argument("--out-features-dir", default="/tmp/项目/repo/MS-TCN2/data/imu_sota/features_xgb")
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    in_features = data_root / "features"
    prob_dir = Path(args.prob_dir)
    out_dir = Path(args.out_features_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"in_features": str(in_features), "prob_dir": str(prob_dir), "out_features": str(out_dir)}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    done = 0
    for fp in sorted(in_features.glob("HNU*.npy")):
        uid = fp.stem
        pp = prob_dir / f"{uid}.npy"
        if not pp.exists():
            continue
        feats = np.load(fp).astype(np.float32, copy=False)  # (D,T)
        prob = np.load(pp).astype(np.float32, copy=False)  # (C,T)
        T = min(int(feats.shape[1]), int(prob.shape[1]))
        if T <= 0:
            continue
        feats = feats[:, :T]
        prob = prob[:, :T]
        logp = np.log(np.clip(prob, float(args.eps), 1.0)).astype(np.float32, copy=False)
        out = np.concatenate([feats, logp], axis=0).astype(np.float32, copy=False)
        np.save(out_dir / f"{uid}.npy", out)
        done += 1

    print(f"built_users={done}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()


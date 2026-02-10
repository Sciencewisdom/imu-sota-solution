#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Infer XGB per-window probabilities on MS-TCN2 features and save per-user npy.")
    ap.add_argument("--data-root", default="/tmp/项目/repo/MS-TCN2/data/imu_sota")
    ap.add_argument("--split", default="4")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default="/tmp/项目/repo/MS-TCN2/data/imu_sota/xgb_prob")
    ap.add_argument("--which", choices=["train", "test", "all"], default="all")
    args = ap.parse_args()

    import xgboost as xgb

    data_root = Path(args.data_root)
    features_dir = data_root / "features"
    splits_dir = data_root / "splits"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bst = xgb.Booster()
    bst.load_model(args.model)

    users = set()
    if args.which in ("train", "all"):
        for line in (splits_dir / f"train.split{args.split}.bundle").read_text(encoding="utf-8").splitlines():
            if line.strip():
                users.add(line.strip().split(".")[0])
    if args.which in ("test", "all"):
        for line in (splits_dir / f"test.split{args.split}.bundle").read_text(encoding="utf-8").splitlines():
            if line.strip():
                users.add(line.strip().split(".")[0])

    meta = {"data_root": str(data_root), "split": args.split, "model": args.model, "which": args.which}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    done = 0
    for uid in sorted(users):
        fp = features_dir / f"{uid}.npy"
        if not fp.exists():
            continue
        feats = np.load(fp).astype(np.float32, copy=False)  # (D,T)
        X = feats.T.copy()  # (T,D)
        prob = bst.predict(xgb.DMatrix(X)).astype(np.float32, copy=False)  # (T,C)
        np.save(out_dir / f"{uid}.npy", prob.T)  # (C,T) for concat convenience
        done += 1

    print(f"inferred_users={done}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()


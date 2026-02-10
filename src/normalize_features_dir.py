#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Per-user feature normalization: z-score each channel over time.")
    ap.add_argument("--in-dir", required=True, help="Input features dir (HNU*.npy as (D,T)).")
    ap.add_argument("--out-dir", required=True, help="Output dir.")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--clip", type=float, default=10.0, help="Clip z-score range (0 disables).")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"in_dir": str(in_dir), "eps": float(args.eps), "clip": float(args.clip)}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    done = 0
    for fp in sorted(in_dir.glob("HNU*.npy")):
        x = np.load(fp).astype(np.float32, copy=False)  # (D,T)
        if x.ndim != 2:
            continue
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True)
        x = (x - mu) / (sd + float(args.eps))
        if float(args.clip) and float(args.clip) > 0:
            x = np.clip(x, -float(args.clip), float(args.clip))
        np.save(out_dir / fp.name, x.astype(np.float32, copy=False))
        done += 1

    print(f"normalized_users={done}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()


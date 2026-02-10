#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _normalize_prob(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1.0).astype(np.float32, copy=False)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return p.astype(np.float32, copy=False)


def _fuse_prob_avg(p1: np.ndarray, p2: np.ndarray, alpha: float) -> np.ndarray:
    a = float(alpha)
    return _normalize_prob((1.0 - a) * p1 + a * p2)


def _fuse_logit_avg(p1: np.ndarray, p2: np.ndarray, alpha: float) -> np.ndarray:
    # Average in log-prob space ~= average logits (up to a constant).
    a = float(alpha)
    l1 = np.log(np.clip(p1, 1e-8, 1.0))
    l2 = np.log(np.clip(p2, 1e-8, 1.0))
    l = (1.0 - a) * l1 + a * l2
    l = l - np.max(l, axis=1, keepdims=True)
    p = np.exp(l).astype(np.float32, copy=False)
    return _normalize_prob(p)


def main():
    ap = argparse.ArgumentParser(description="Fuse two infer-cache dirs (npz) into one.")
    ap.add_argument("--cache-a", required=True)
    ap.add_argument("--cache-b", required=True)
    ap.add_argument("--alpha", type=float, required=True, help="weight for cache-b (0..1)")
    ap.add_argument("--mode", choices=["prob_avg", "logit_avg"], default="logit_avg")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ca = Path(args.cache_a)
    cb = Path(args.cache_b)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "cache_a": str(ca),
        "cache_b": str(cb),
        "alpha": float(args.alpha),
        "mode": str(args.mode),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    done = 0
    for pa in sorted(ca.glob("HNU*.npz")):
        uid = pa.stem
        pb = cb / f"{uid}.npz"
        if not pb.exists():
            continue
        za = np.load(pa)
        zb = np.load(pb)
        sa = za["starts"].astype(np.int64, copy=False)
        sb = zb["starts"].astype(np.int64, copy=False)
        p1 = za["prob"].astype(np.float32, copy=False)
        p2 = zb["prob"].astype(np.float32, copy=False)
        if p1.ndim != 2 or p2.ndim != 2:
            continue
        T = min(int(sa.shape[0]), int(sb.shape[0]), int(p1.shape[0]), int(p2.shape[0]))
        if T <= 0:
            continue
        # Require aligned timestamps for safety.
        if not np.array_equal(sa[:T], sb[:T]):
            continue
        sa = sa[:T].copy()
        p1 = p1[:T].copy()
        p2 = p2[:T].copy()

        if args.mode == "prob_avg":
            pf = _fuse_prob_avg(p1, p2, float(args.alpha))
        else:
            pf = _fuse_logit_avg(p1, p2, float(args.alpha))
        np.savez_compressed(out / f"{uid}.npz", starts=sa, prob=pf)
        done += 1

    print(f"fused_users={done}")
    print(f"out_dir={out}")


if __name__ == "__main__":
    main()


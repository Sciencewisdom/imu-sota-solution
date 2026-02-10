#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Build infer_cache-style npz from prob (C,T) + timestamps (T,) for tuning.")
    ap.add_argument("--prob-dir", required=True, help="Directory with per-user prob .npy (C,T).")
    ap.add_argument("--timestamps-dir", required=True, help="Directory with per-user starts .npy (T,).")
    ap.add_argument("--user-list", required=True, help="File with one user per line (stems).")
    ap.add_argument("--labels-json", default="", help="Optional labels list json (written to meta.json).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    prob_dir = Path(args.prob_dir)
    ts_dir = Path(args.timestamps_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = None
    if args.labels_json:
        obj = json.loads(Path(args.labels_json).read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "labels" in obj:
            labels = obj["labels"]
        elif isinstance(obj, list):
            labels = obj

    meta = {"prob_dir": str(prob_dir), "timestamps_dir": str(ts_dir), "labels": labels}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    users = [x.strip() for x in Path(args.user_list).read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
    done = 0
    for uid in users:
        pp = prob_dir / f"{uid}.npy"
        tp = ts_dir / f"{uid}.npy"
        if not pp.exists() or not tp.exists():
            continue
        prob = np.load(pp).astype(np.float32, copy=False)  # (C,T)
        starts = np.load(tp).astype(np.int64, copy=False)  # (T,)
        if prob.ndim != 2 or starts.ndim != 1:
            continue
        T = min(int(prob.shape[1]), int(starts.shape[0]))
        if T <= 0:
            continue
        prob_TC = prob[:, :T].T.copy()  # (T,C)
        starts = starts[:T].copy()
        np.savez_compressed(out_dir / f"{uid}.npz", starts=starts, prob=prob_TC)
        done += 1

    print(f"cached_users={done}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()


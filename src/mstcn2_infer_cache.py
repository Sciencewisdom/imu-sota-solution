#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Reuse postprocess + feature utilities from /root.
if "/root" not in sys.path:
    sys.path.insert(0, "/root")


def _load_mapping(mapping_txt: Path):
    idx_to_cat = {}
    for line in mapping_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        a, b = line.split(maxsplit=1)
        idx_to_cat[int(a)] = b.strip()
    return idx_to_cat


def main():
    ap = argparse.ArgumentParser(description="Run MS-TCN2 model and cache per-window probabilities (npz).")
    ap.add_argument("--repo", default="/tmp/项目/repo/MS-TCN2", help="Path to MS-TCN2 repo root.")
    ap.add_argument("--dataset", default="imu")
    ap.add_argument("--split", default="1")
    ap.add_argument("--model-dataset", default="", help="Dataset name used for model_dir (defaults to --dataset).")
    ap.add_argument("--epoch", type=int, required=True, help="Epoch number to load (epoch-N.model).")
    ap.add_argument("--num-f-maps", type=int, default=256, help="Model width (must match training).")
    ap.add_argument("--num-layers-pg", type=int, default=12, help="PG layers (must match training).")
    ap.add_argument("--num-layers-r", type=int, default=12, help="R layers (must match training).")
    ap.add_argument("--num-r", type=int, default=5, help="Number of refinement stages (must match training).")
    ap.add_argument("--features-dir", default="", help="Override features directory (expects *.npy as (D,T)).")
    ap.add_argument("--timestamps-dir", default="", help="Override timestamps directory (expects *.npy as (T,) starts).")
    ap.add_argument("--glob", default="HNU*.npy", help="Feature glob under features-dir (default: HNU*.npy).")
    ap.add_argument("--user-list", default="", help="Optional file with one user per line (stem only).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    repo = Path(args.repo)
    # Ensure MS-TCN2 python modules (model.py, etc.) are importable.
    repo_str = str(repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    data_root = repo / "data" / args.dataset
    features_dir = Path(args.features_dir) if args.features_dir else (data_root / "features")
    ts_dir = Path(args.timestamps_dir) if args.timestamps_dir else (data_root / "timestamps")
    mapping = _load_mapping(data_root / "mapping.txt")
    labels = [mapping[i] for i in sorted(mapping)]

    model_dataset = args.model_dataset or args.dataset
    model_dir = repo / "models" / model_dataset / f"split_{args.split}"
    model_path = model_dir / f"epoch-{int(args.epoch)}.model"
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    import torch
    from model import MS_TCN2

    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # These must match training hyperparams.
    # Read from a small json if present; else fallback to CLI defaults.
    hp_path = model_dir / "imu_hparams.json"
    if hp_path.exists():
        hp = json.loads(hp_path.read_text(encoding="utf-8"))
    else:
        hp = {
            "num_layers_PG": int(args.num_layers_pg),
            "num_layers_R": int(args.num_layers_r),
            "num_R": int(args.num_r),
            "num_f_maps": int(args.num_f_maps),
        }

    feats_dim = int(np.load(next(features_dir.glob("*.npy"))).shape[0])
    n_classes = int(len(labels))
    net = MS_TCN2(
        int(hp["num_layers_PG"]),
        int(hp["num_layers_R"]),
        int(hp["num_R"]),
        int(hp["num_f_maps"]),
        feats_dim,
        n_classes,
    ).to(dev)
    net.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    net.eval()

    allow = None
    if args.user_list:
        allow = set(
            x.strip()
            for x in Path(args.user_list).read_text(encoding="utf-8", errors="ignore").splitlines()
            if x.strip()
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "repo": str(repo),
                "dataset": args.dataset,
                "split": args.split,
                "epoch": int(args.epoch),
                "labels": labels,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    done = 0
    for fp in sorted(features_dir.glob(str(args.glob))):
        uid = fp.stem
        if allow is not None and uid not in allow:
            continue
        tp = ts_dir / f"{uid}.npy"
        if not tp.exists():
            continue
        feats = np.load(fp).astype(np.float32, copy=False)  # (D, T)
        starts = np.load(tp).astype(np.int64, copy=False)  # (T,)
        if feats.ndim != 2 or starts.ndim != 1 or feats.shape[1] != starts.shape[0]:
            continue

        x = torch.from_numpy(feats).unsqueeze(0).to(dev)  # (1, D, T)
        with torch.no_grad():
            out = net(x)  # (S, 1, C, T)
            logits = out[-1, 0]  # (C, T)
            prob = torch.softmax(logits, dim=0).transpose(0, 1).contiguous().cpu().numpy()  # (T, C)

        np.savez_compressed(out_dir / f"{uid}.npz", starts=starts, prob=prob.astype(np.float32, copy=False))
        done += 1

    print(f"cached_users={done}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()

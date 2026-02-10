#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_mapping(mapping_txt: Path):
    idx_to_cat = {}
    for line in mapping_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        a, b = line.split(maxsplit=1)
        idx_to_cat[int(a)] = b.strip()
    return idx_to_cat


def _load_labels(gt_txt: Path, actions_dict: dict[str, int]):
    y = []
    for line in gt_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        lab = line.strip()
        if not lab:
            continue
        y.append(int(actions_dict[lab]))
    return np.asarray(y, dtype=np.int16)


def main():
    ap = argparse.ArgumentParser(description="Train XGBoost window classifier on MS-TCN2 prepared window features.")
    ap.add_argument("--data-root", default="/tmp/项目/repo/MS-TCN2/data/imu_sota")
    ap.add_argument("--split", default="4")
    ap.add_argument("--max-per-class", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out-model", default="/tmp/项目/repo/imu_mstcn2/xgb_window.ubj")
    ap.add_argument("--out-label-map", default="/tmp/项目/repo/imu_mstcn2/label_map.json")
    ap.add_argument("--out-report", default="/tmp/项目/repo/imu_mstcn2/xgb_train_report.json")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    features_dir = data_root / "features"
    gt_dir = data_root / "groundTruth"
    splits_dir = data_root / "splits"
    mapping = _load_mapping(data_root / "mapping.txt")
    labels = [mapping[i] for i in sorted(mapping)]
    actions_dict = {lab: i for i, lab in enumerate(labels)}

    train_list = (splits_dir / f"train.split{args.split}.bundle").read_text(encoding="utf-8").splitlines()
    train_list = [x.strip() for x in train_list if x.strip()]

    Xs = []
    ys = []
    groups = []
    for vid in train_list:
        uid = vid.split(".")[0]
        fp = features_dir / f"{uid}.npy"
        gp = gt_dir / f"{uid}.txt"
        if not fp.exists() or not gp.exists():
            continue
        feats = np.load(fp).astype(np.float32, copy=False)  # (D, T)
        y = _load_labels(gp, actions_dict)  # (T,)
        T = min(int(feats.shape[1]), int(y.shape[0]))
        if T <= 0:
            continue
        Xs.append(feats[:, :T].T.copy())  # (T,D)
        ys.append(y[:T].copy())
        groups.append(np.full((T,), uid))

    if not Xs:
        raise SystemExit("No training samples found.")
    X = np.vstack(Xs)
    y = np.concatenate(ys).astype(np.int32, copy=False)
    groups = np.concatenate(groups)

    rng = np.random.default_rng(int(args.seed))

    def cap_per_class(Xm, ym, max_n: int):
        if max_n <= 0:
            return Xm, ym
        keep = []
        for c in np.unique(ym):
            idx = np.where(ym == c)[0]
            if idx.shape[0] > max_n:
                idx = rng.choice(idx, size=max_n, replace=False)
            keep.append(idx)
        keep = np.concatenate(keep)
        rng.shuffle(keep)
        return Xm[keep], ym[keep]

    X, y = cap_per_class(X, y, int(args.max_per_class))

    import xgboost as xgb

    n_classes = int(len(labels))
    params = dict(
        n_estimators=1200,
        max_depth=10,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=int(args.seed),
        n_jobs=0,
        tree_method="hist",
        device="cuda" if args.gpu else "cpu",
    )
    clf = xgb.XGBClassifier(**params)
    clf.fit(X, y)

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    clf.get_booster().save_model(str(out_model))

    out_label = Path(args.out_label_map)
    out_label.parent.mkdir(parents=True, exist_ok=True)
    out_label.write_text(json.dumps({i: labels[i] for i in range(len(labels))}, ensure_ascii=False, indent=2), encoding="utf-8")

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(
        json.dumps(
            {
                "data_root": str(data_root),
                "split": str(args.split),
                "n_classes": n_classes,
                "labels": labels,
                "train_samples": int(X.shape[0]),
                "max_per_class": int(args.max_per_class),
                "gpu": bool(args.gpu),
                "model_params": {k: params[k] for k in params if k not in ("n_jobs",)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"saved_model={out_model}")
    print(f"saved_label_map={out_label}")
    print(f"train_samples={X.shape[0]}")


if __name__ == "__main__":
    main()


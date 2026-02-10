#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Reuse existing fast IMU featurization in /root.
if "/root" not in sys.path:
    sys.path.insert(0, "/root")


def _is_tsv_sensor(p: Path) -> bool:
    try:
        return p.read_bytes()[:16].startswith(b"ACC_TIME")
    except Exception:
        return False


def _split_sessions(ts: np.ndarray, Xc: np.ndarray):
    # Split by very large timestamp gaps using the same heuristic as trim_time_series,
    # but keep all sessions to avoid dropping labeled segments.
    if ts.size == 0:
        return []
    if ts.shape[0] < 200:
        return [(ts, Xc)]
    dt = np.diff(ts)
    pos = dt[dt > 0]
    if pos.size == 0:
        return [(ts, Xc)]
    med = float(np.median(pos))
    if med <= 0:
        return [(ts, Xc)]
    big = np.where(dt > (med * 1000.0))[0]
    if big.size == 0:
        return [(ts, Xc)]
    cuts = np.concatenate(([0], big + 1, [ts.shape[0]]))
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        a = int(a)
        b = int(b)
        if b - a <= 0:
            continue
        out.append((ts[a:b], Xc[a:b]))
    return out


def _one_user(task):
    # Separate function for multiprocessing pickling.
    uid, txt_path, seg_list, cat_to_idx, window_ms, stride_ms, min_points, cover_th, trim_mode = task
    import train_tsv_baseline as fe

    try:
        p = Path(txt_path)
        if not _is_tsv_sensor(p):
            return uid, None

        df = fe._read_tsv_sensor(p).dropna(subset=["ACC_TIME"] + fe.CHANNELS)
        if df.empty:
            return uid, None

        ts = df["ACC_TIME"].astype("int64", copy=False).to_numpy()
        Xc = df[fe.CHANNELS].astype("float32").to_numpy()
        if ts.shape[0] >= 2 and np.any(ts[1:] < ts[:-1]):
            order = np.argsort(ts, kind="mergesort")
            ts = ts[order]
            Xc = Xc[order]
        if ts.size == 0:
            return uid, None

        sessions = _split_sessions(ts, Xc)
        if not sessions:
            return uid, None
        if str(trim_mode) == "longest":
            # Keep only the longest session (legacy behavior).
            sessions = [max(sessions, key=lambda x: int(x[0].shape[0]))]

        starts_all = []
        feats_all = []
        y_all = []

        for ts_s, Xc_s in sessions:
            if ts_s.size == 0:
                continue
            tmin = int(ts_s.min())
            tmax = int(ts_s.max())
            if tmax - tmin < int(window_ms):
                continue
            n_total = int((tmax - tmin - int(window_ms)) // int(stride_ms)) + 1
            if n_total <= 0:
                continue
            starts = (tmin + (np.arange(n_total, dtype=np.int64) * int(stride_ms))).astype(np.int64, copy=False)
            ends = starts + int(window_ms)
            l = np.searchsorted(ts_s, starts, side="left")
            r = np.searchsorted(ts_s, ends, side="left")
            valid = (r > l) & ((r - l) >= int(min_points))
            if not np.any(valid):
                continue

            starts_v = starts[valid]
            l_v = l[valid]
            r_v = r[valid]
            y = fe.label_windows(
                starts_v,
                window_ms=int(window_ms),
                seg_list=seg_list,
                cat_to_idx=cat_to_idx,
                cover_th=float(cover_th),
            )
            X = np.empty((starts_v.shape[0], fe.N_FEATS), dtype=np.float32)
            for i in range(starts_v.shape[0]):
                li = int(l_v[i])
                ri = int(r_v[i])
                X[i] = fe.featurize_window(Xc_s[li:ri], ts_s[li:ri])

            starts_all.append(starts_v)
            feats_all.append(X)
            y_all.append(y)

        if not starts_all:
            return uid, None

        starts_v = np.concatenate(starts_all, axis=0).astype(np.int64, copy=False)
        X = np.concatenate(feats_all, axis=0).astype(np.float32, copy=False)
        y = np.concatenate(y_all, axis=0).astype(np.int16, copy=False)

        # Return in MS-TCN2 expected orientation: (D, T)
        feats = X.T.copy()
        return uid, {"starts": starts_v, "feats": feats, "y": y}
    except Exception:
        return uid, None


def main():
    ap = argparse.ArgumentParser(description="Prepare IMU dataset in MS-TCN2 format (features/.npy + groundTruth/.txt + splits).")
    ap.add_argument("--train-dir", default="/root/data/train_data")
    ap.add_argument("--label-file", default="/root/csv_files/赛题2金标-训练集_Sheet1.csv")
    ap.add_argument("--id-alias-file", default="/root/csv_files/赛题2异常数据说明-训练集_训练集.csv")
    ap.add_argument("--out-root", default="/tmp/项目/repo/MS-TCN2/data/imu")
    ap.add_argument("--split", default="1")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window-ms", type=int, default=2560)
    ap.add_argument("--stride-ms", type=int, default=640)
    ap.add_argument("--min-points", type=int, default=30)
    ap.add_argument("--cover-th", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--trim", choices=["longest", "none"], default="longest", help="Session handling strategy. 'none' keeps all sessions.")
    args = ap.parse_args()

    import train_tsv_baseline as fe
    from sklearn.model_selection import GroupShuffleSplit

    train_dir = Path(args.train_dir)
    segs, cats = fe.load_segments(Path(args.label_file))
    segs = fe.apply_aliases(segs, fe.load_id_aliases(Path(args.id_alias_file)))

    labels = ["background"] + list(cats)
    cat_to_idx = {c: i for i, c in enumerate(labels)}

    # Only users that have labels and data file.
    files = []
    for p in sorted(train_dir.glob("HNU*.txt")):
        if p.stem in segs and _is_tsv_sensor(p):
            files.append(p)

    if not files:
        raise SystemExit("No matching HNU*.txt found under --train-dir that also exist in labels.")

    users = [p.stem for p in files]
    gss = GroupShuffleSplit(n_splits=1, test_size=float(args.test_size), random_state=int(args.seed))
    tr_idx, te_idx = next(gss.split(users, users, groups=users))
    tr_users = [users[i] for i in tr_idx]
    te_users = [users[i] for i in te_idx]

    out_root = Path(args.out_root)
    features_dir = out_root / "features"
    gt_dir = out_root / "groundTruth"
    splits_dir = out_root / "splits"
    ts_dir = out_root / "timestamps"
    for d in (features_dir, gt_dir, splits_dir, ts_dir):
        d.mkdir(parents=True, exist_ok=True)

    # mapping.txt: "idx label"
    (out_root / "mapping.txt").write_text("\n".join([f"{i} {c}" for i, c in enumerate(labels)]) + "\n", encoding="utf-8")

    # splits: entries are gt filenames (with .txt extension)
    train_bundle = splits_dir / f"train.split{args.split}.bundle"
    test_bundle = splits_dir / f"test.split{args.split}.bundle"
    train_bundle.write_text("\n".join([f"{u}.txt" for u in tr_users]) + "\n", encoding="utf-8")
    test_bundle.write_text("\n".join([f"{u}.txt" for u in te_users]) + "\n", encoding="utf-8")

    # meta (for later inference/postprocess)
    meta = {
        "labels": labels,
        "window_ms": int(args.window_ms),
        "stride_ms": int(args.stride_ms),
        "min_points": int(args.min_points),
        "cover_th": float(args.cover_th),
        "train_users": tr_users,
        "test_users": te_users,
    }
    (out_root / "imu_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    tasks = []
    for p in files:
        tasks.append(
            (
                p.stem,
                str(p),
                segs[p.stem],
                cat_to_idx,
                int(args.window_ms),
                int(args.stride_ms),
                int(args.min_points),
                float(args.cover_th),
                str(args.trim),
            )
        )

    done = 0
    skipped = 0
    if int(args.workers) <= 1:
        for t in tasks:
            uid, out = _one_user(t)
            if out is None:
                skipped += 1
                continue
            starts = out["starts"].astype(np.int64, copy=False)
            feats = out["feats"].astype(np.float32, copy=False)  # (D,T)
            y = out["y"].astype(np.int16, copy=False)

            np.save(features_dir / f"{uid}.npy", feats)
            (gt_dir / f"{uid}.txt").write_text("\n".join([labels[int(i)] for i in y.tolist()]) + "\n", encoding="utf-8")
            np.save(ts_dir / f"{uid}.npy", starts)
            done += 1
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = [ex.submit(_one_user, t) for t in tasks]
            for fut in as_completed(futs):
                uid, out = fut.result()
                if out is None:
                    skipped += 1
                    continue
                starts = out["starts"].astype(np.int64, copy=False)
                feats = out["feats"].astype(np.float32, copy=False)  # (D,T)
                y = out["y"].astype(np.int16, copy=False)

                np.save(features_dir / f"{uid}.npy", feats)
                (gt_dir / f"{uid}.txt").write_text("\n".join([labels[int(i)] for i in y.tolist()]) + "\n", encoding="utf-8")
                np.save(ts_dir / f"{uid}.npy", starts)
                done += 1

    print(f"prepared_users={done} skipped={skipped}")
    print(f"out_root={out_root}")


if __name__ == "__main__":
    main()

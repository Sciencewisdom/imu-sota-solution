#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

if "/root" not in sys.path:
    sys.path.insert(0, "/root")


def _is_tsv_sensor(p: Path) -> bool:
    try:
        return p.read_bytes()[:16].startswith(b"ACC_TIME")
    except Exception:
        return False


def _split_sessions(ts: np.ndarray, Xc: np.ndarray):
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
    uid, txt_path, window_ms, stride_ms, min_points, trim_mode = task
    try:
        import train_tsv_baseline as fe

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
            sessions = [max(sessions, key=lambda x: int(x[0].shape[0]))]

        starts_all = []
        feats_all = []
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
            X = np.empty((starts_v.shape[0], fe.N_FEATS), dtype=np.float32)
            for i in range(starts_v.shape[0]):
                li = int(l_v[i])
                ri = int(r_v[i])
                X[i] = fe.featurize_window(Xc_s[li:ri], ts_s[li:ri])
            starts_all.append(starts_v)
            feats_all.append(X)

        if not starts_all:
            return uid, None

        starts_v = np.concatenate(starts_all, axis=0).astype(np.int64, copy=False)
        X = np.concatenate(feats_all, axis=0).astype(np.float32, copy=False)
        feats = X.T.copy()  # (D,T)
        return uid, {"starts": starts_v, "feats": feats}
    except Exception:
        return uid, None


def main():
    ap = argparse.ArgumentParser(description="Prepare MS-TCN2 features/timestamps only (no labels).")
    ap.add_argument("--in-dir", required=True, help="Directory containing raw sensor .txt files.")
    ap.add_argument("--glob", default="HNU*.txt", help="Input glob pattern (default: HNU*.txt).")
    ap.add_argument("--recursive", action="store_true", help="Recursively scan --in-dir.")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--split", default="1")
    ap.add_argument("--window-ms", type=int, default=2560)
    ap.add_argument("--stride-ms", type=int, default=640)
    ap.add_argument("--min-points", type=int, default=30)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--trim", choices=["longest", "none"], default="none")
    ap.add_argument("--labels-json", required=True, help="Path to json containing labels list (e.g. imu_meta.json).")
    args = ap.parse_args()

    labels_obj = json.loads(Path(args.labels_json).read_text(encoding="utf-8"))
    labels = labels_obj["labels"] if isinstance(labels_obj, dict) and "labels" in labels_obj else labels_obj
    labels = [str(x) for x in labels]

    in_dir = Path(args.in_dir)
    if args.recursive:
        files = [p for p in sorted(in_dir.rglob(str(args.glob))) if p.is_file() and _is_tsv_sensor(p)]
    else:
        files = [p for p in sorted(in_dir.glob(str(args.glob))) if p.is_file() and _is_tsv_sensor(p)]
    if not files:
        raise SystemExit(f"No usable files found under {in_dir} with glob={args.glob} (recursive={bool(args.recursive)}).")

    out_root = Path(args.out_root)
    features_dir = out_root / "features"
    splits_dir = out_root / "splits"
    ts_dir = out_root / "timestamps"
    for d in (features_dir, splits_dir, ts_dir):
        d.mkdir(parents=True, exist_ok=True)

    (out_root / "mapping.txt").write_text("\n".join([f"{i} {c}" for i, c in enumerate(labels)]) + "\n", encoding="utf-8")
    (out_root / "meta.json").write_text(
        json.dumps(
            {
                "labels": labels,
                "window_ms": int(args.window_ms),
                "stride_ms": int(args.stride_ms),
                "min_points": int(args.min_points),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # For MS-TCN2 predict, we only need test bundle list.
    (splits_dir / f"test.split{args.split}.bundle").write_text(
        "\n".join([f"{p.stem}.txt" for p in files]) + "\n", encoding="utf-8"
    )
    (splits_dir / f"train.split{args.split}.bundle").write_text("", encoding="utf-8")

    tasks = [(p.stem, str(p), int(args.window_ms), int(args.stride_ms), int(args.min_points), str(args.trim)) for p in files]
    done = 0
    skipped = 0
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [ex.submit(_one_user, t) for t in tasks]
        for fut in as_completed(futs):
            uid, out = fut.result()
            if out is None:
                skipped += 1
                continue
            np.save(features_dir / f"{uid}.npy", out["feats"].astype(np.float32, copy=False))
            np.save(ts_dir / f"{uid}.npy", out["starts"].astype(np.int64, copy=False))
            done += 1

    print(f"prepared_users={done} skipped={skipped}")
    print(f"out_root={out_root}")


if __name__ == "__main__":
    main()

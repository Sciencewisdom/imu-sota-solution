#!/usr/bin/env python3
from __future__ import annotations

"""
Competition interface runner.

Reads hidden test .txt files via the organizer-provided DataReader (./test_data),
runs our MS-TCN2 model, applies tuned postprocess config, and writes submission.xlsx
via organizer-provided DataOutput.

This script is designed to be pyinstaller-packable. It avoids relying on external
repo working directories by using absolute paths by default (can be overridden by CLI).
"""

import argparse
import json
import sys
from pathlib import Path

import math

import numpy as np


def _ensure_sys_path(p: Path):
    s = str(p.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def _is_tsv_sensor_text(text: str) -> bool:
    return text.startswith("ACC_TIME")


def _parse_tsv_sensor_text(text: str):
    # Parse only the columns we need (ACC_TIME + ACC/GYRO) for speed/memory.
    import pandas as pd
    from io import StringIO

    CHANNELS = ["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"]
    usecols = ["ACC_TIME"] + CHANNELS
    df = pd.read_csv(StringIO(text), sep="\t", usecols=usecols, engine="c", low_memory=False)
    for c in usecols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _read_test_dir_robust(test_data_dir: Path) -> dict[str, str]:
    # Organizer sample uses utf-8; in practice some files may not be strict utf-8.
    out: dict[str, str] = {}
    for p in sorted(test_data_dir.glob("*.txt")):
        if not p.is_file():
            continue
        b = p.read_bytes()
        text = None
        # 1) utf-8 strict 2) gb18030 strict 3) utf-8 ignore
        for mode in (("utf-8", "strict"), ("gb18030", "strict"), ("utf-8", "ignore")):
            enc, err = mode
            try:
                text = b.decode(enc, errors=err)
                break
            except Exception:
                text = None
        if text is None:
            continue
        out[p.stem] = text
    return out


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


def _build_windows_for_user(text: str, window_ms: int, stride_ms: int, min_points: int):
    if not _is_tsv_sensor_text(text):
        return None

    CHANNELS = ["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"]
    df = _parse_tsv_sensor_text(text).dropna(subset=["ACC_TIME"] + CHANNELS)
    if df.empty:
        return None

    ts = df["ACC_TIME"].astype("int64", copy=False).to_numpy()
    Xc = df[CHANNELS].astype("float32").to_numpy()
    if ts.shape[0] >= 2 and np.any(ts[1:] < ts[:-1]):
        order = np.argsort(ts, kind="mergesort")
        ts = ts[order]
        Xc = Xc[order]

    sessions = _split_sessions(ts, Xc)
    if not sessions:
        return None

    # Feature spec must match training: N_FEATS=58.
    def _estimate_fs_hz(tw_ms: np.ndarray) -> float:
        if tw_ms.size < 4:
            return 0.0
        dt = np.diff(tw_ms.astype(np.int64, copy=False))
        dt = dt[dt > 0]
        med = float(np.median(dt)) if dt.size else 0.0
        return (1000.0 / med) if med > 0 else 0.0

    def _fft_feats(x: np.ndarray, fs_hz: float):
        # Fixed 8 dims: band energies (4) + peak_f + peak_p + entropy + total_power
        if x.size < 16 or not np.isfinite(fs_hz) or fs_hz <= 0:
            return [0.0] * 8
        x = x.astype(np.float32, copy=False)
        x = x - float(x.mean())
        n = int(x.size)
        spec = np.fft.rfft(x)
        pwr = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32, copy=False)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
        if pwr.size <= 1:
            return [0.0] * 8
        total = float(pwr[1:].sum())
        if total <= 0:
            return [0.0] * 8

        def band(lo, hi):
            m = (freqs >= lo) & (freqs < hi)
            v = float(pwr[m].sum())
            return v / total

        b0 = band(0.5, 3.0)
        b1 = band(3.0, 6.0)
        b2 = band(6.0, 10.0)
        b3 = band(10.0, 20.0)
        k = int(np.argmax(pwr[1:])) + 1
        peak_f = float(freqs[k])
        peak_p = float(pwr[k] / total)
        p = (pwr[1:] / total).astype(np.float32, copy=False)
        p = p[p > 0]
        ent = float(-(p * np.log2(p)).sum()) if p.size else 0.0
        return [b0, b1, b2, b3, peak_f, peak_p, ent, total]

    def _featurize_window(v: np.ndarray, tw_ms: np.ndarray) -> np.ndarray:
        # Must match /root/train_tsv_baseline.py featurize_window() output ordering/length.
        mean = v.mean(axis=0)
        std = v.std(axis=0)
        vmin = v.min(axis=0)
        vmax = v.max(axis=0)
        q25, q75 = np.percentile(v, [25, 75], axis=0)
        N_FFT_FEATS_PER_SERIES = 8
        N_FEATS = (6 * len(CHANNELS)) + 6 + (2 * N_FFT_FEATS_PER_SERIES)  # 58
        row = np.empty((N_FEATS,), dtype=np.float32)
        off = 0
        for arr in (mean, std, vmin, vmax, q25, q75):
            row[off : off + len(CHANNELS)] = arr.astype(np.float32, copy=False)
            off += len(CHANNELS)
        acc = v[:, 0:3]
        gyr = v[:, 3:6]
        acc_mag = np.sqrt((acc * acc).sum(axis=1))
        gyr_mag = np.sqrt((gyr * gyr).sum(axis=1))
        row[off + 0] = float(acc_mag.mean())
        row[off + 1] = float(acc_mag.std())
        row[off + 2] = float(acc_mag.max())
        row[off + 3] = float(gyr_mag.mean())
        row[off + 4] = float(gyr_mag.std())
        row[off + 5] = float(gyr_mag.max())
        off += 6
        fs = _estimate_fs_hz(tw_ms)
        row[off : off + N_FFT_FEATS_PER_SERIES] = np.asarray(_fft_feats(acc_mag, fs), dtype=np.float32)
        off += N_FFT_FEATS_PER_SERIES
        row[off : off + N_FFT_FEATS_PER_SERIES] = np.asarray(_fft_feats(gyr_mag, fs), dtype=np.float32)
        return row

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
        X = np.empty((starts_v.shape[0], 58), dtype=np.float32)
        for i in range(starts_v.shape[0]):
            li = int(l_v[i])
            ri = int(r_v[i])
            X[i] = _featurize_window(Xc_s[li:ri], ts_s[li:ri])
        starts_all.append(starts_v)
        feats_all.append(X)

    if not starts_all:
        return None

    starts_v = np.concatenate(starts_all, axis=0).astype(np.int64, copy=False)  # (T,)
    Xw = np.concatenate(feats_all, axis=0).astype(np.float32, copy=False)  # (T, D)
    feats = Xw.T.copy()  # (D, T)
    return starts_v, feats


def _normalize_feats_user(feats_dt: np.ndarray, clip: float = 10.0) -> np.ndarray:
    # feats_dt: (D, T)
    x = feats_dt.astype(np.float32, copy=False)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0).astype(np.float32, copy=False)
    z = (x - mu) / sd
    z = np.clip(z, -float(clip), float(clip)).astype(np.float32, copy=False)
    return z


def _load_mapping(mapping_txt: Path):
    idx_to_cat = {}
    for line in mapping_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        a, b = line.split(maxsplit=1)
        idx_to_cat[int(a)] = b.strip()
    labels = [idx_to_cat[i] for i in sorted(idx_to_cat)]
    return idx_to_cat, labels


def _apply_temperature(prob: np.ndarray, temp: float) -> np.ndarray:
    t = float(temp)
    if not np.isfinite(t) or t <= 0 or abs(t - 1.0) < 1e-6:
        return prob
    p = np.clip(prob, 1e-8, 1.0).astype(np.float32, copy=False)
    p = p ** (1.0 / t)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return p.astype(np.float32, copy=False)


def _smooth_prob(prob: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return prob
    k = int(k)
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / float(k)
    out = np.empty_like(prob, dtype=np.float32)
    for c in range(prob.shape[1]):
        out[:, c] = np.convolve(prob[:, c], kernel, mode="same")
    return out


def _build_enter_exit_thresholds(idx_to_cat: dict[int, str], enter_default: float, class_enter: dict[str, float], hyst_delta: float, exit_floor: float):
    enter = {}
    for _, cat in idx_to_cat.items():
        if cat == "background":
            enter[cat] = 1e9
        else:
            enter[cat] = float(class_enter.get(cat, enter_default))
    exit_th = {}
    for cat, th in enter.items():
        if cat == "background":
            exit_th[cat] = 1e9
        else:
            exit_th[cat] = max(float(th) - float(hyst_delta), float(exit_floor))
    return enter, exit_th


def _segments_from_prob_hyst(
    starts: np.ndarray,
    prob: np.ndarray,
    idx_to_cat: dict[int, str],
    cat_to_idx: dict[str, int],
    window_ms: int,
    gap_ms: int,
    margin: float,
    enter_th: dict[str, float],
    exit_th: dict[str, float],
    min_dur_ms,
    seg_score_th,
):
    segs = []
    cur = "background"
    seg_start = None
    seg_end = None
    score_sum = 0.0
    score_cnt = 0
    for i in range(prob.shape[0]):
        t0 = int(starts[i])
        t1 = t0 + int(window_ms)
        if cur != "background" and seg_end is not None and t0 > int(seg_end) + int(gap_ms):
            dur = int(seg_end) - int(seg_start)
            avg = score_sum / max(1, score_cnt)
            mind = int(min_dur_ms.get(cur, 0)) if isinstance(min_dur_ms, dict) else int(min_dur_ms)
            sth = float(seg_score_th.get(cur, 0.0)) if isinstance(seg_score_th, dict) else float(seg_score_th)
            if dur >= mind and avg >= sth:
                segs.append((cur, int(seg_start), int(seg_end), float(avg)))
            cur = "background"
            seg_start = seg_end = None
            score_sum = 0.0
            score_cnt = 0

        top2 = np.argpartition(prob[i], -2)[-2:]
        s0 = float(prob[i, int(top2[0])])
        s1 = float(prob[i, int(top2[1])])
        if s1 >= s0:
            best_idx, best_s, second_s = int(top2[1]), s1, s0
        else:
            best_idx, best_s, second_s = int(top2[0]), s0, s1
        best_cat = idx_to_cat.get(best_idx, "background")

        if cur == "background":
            if best_cat != "background" and best_s >= float(enter_th.get(best_cat, 1e9)) and (best_s - second_s) >= float(margin):
                cur = best_cat
                seg_start = t0
                seg_end = t1
                score_sum = best_s
                score_cnt = 1
        else:
            ci = int(cat_to_idx.get(cur, 0))
            cur_s = float(prob[i, ci]) if 0 <= ci < prob.shape[1] else 0.0
            if cur_s >= float(exit_th.get(cur, 1e9)):
                seg_end = t1
                score_sum += cur_s
                score_cnt += 1
            else:
                dur = int(seg_end) - int(seg_start)
                avg = score_sum / max(1, score_cnt)
                mind = int(min_dur_ms.get(cur, 0)) if isinstance(min_dur_ms, dict) else int(min_dur_ms)
                sth = float(seg_score_th.get(cur, 0.0)) if isinstance(seg_score_th, dict) else float(seg_score_th)
                if dur >= mind and avg >= sth:
                    segs.append((cur, int(seg_start), int(seg_end), float(avg)))
                cur = "background"
                seg_start = seg_end = None
                score_sum = 0.0
                score_cnt = 0
                if best_cat != "background" and best_s >= float(enter_th.get(best_cat, 1e9)) and (best_s - second_s) >= float(margin):
                    cur = best_cat
                    seg_start = t0
                    seg_end = t1
                    score_sum = best_s
                    score_cnt = 1

    if cur != "background" and seg_start is not None and seg_end is not None:
        dur = int(seg_end) - int(seg_start)
        avg = score_sum / max(1, score_cnt)
        mind = int(min_dur_ms.get(cur, 0)) if isinstance(min_dur_ms, dict) else int(min_dur_ms)
        sth = float(seg_score_th.get(cur, 0.0)) if isinstance(seg_score_th, dict) else float(seg_score_th)
        if dur >= mind and avg >= sth:
            segs.append((cur, int(seg_start), int(seg_end), float(avg)))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-data-dir", default="./test_data", help="Directory containing hidden *.txt files.")
    ap.add_argument("--out-xlsx", default="./submission.xlsx")
    ap.add_argument("--repo", default="/tmp/项目/repo/MS-TCN2", help="Fallback MS-TCN2 repo root (used if package files missing).")
    ap.add_argument("--dataset", default="imu_sota_notrim", help="Fallback dataset name (used if mapping.txt missing).")
    ap.add_argument("--split", default="98")
    ap.add_argument("--epoch", type=int, default=79)
    ap.add_argument(
        "--model-paths",
        default="",
        help="Optional comma-separated list of explicit model paths. If set, overrides --dataset/--split/--epoch.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config-json", default="", help="Postprocess config json. Defaults to ./best_cfg.json if present.")
    ap.add_argument("--window-ms", type=int, default=2560)
    ap.add_argument("--stride-ms", type=int, default=640)
    ap.add_argument("--min-points", type=int, default=30)
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent

    # Organizer interface: prefer local copies (packaged), fall back to /root if present.
    _ensure_sys_path(base_dir)
    if (base_dir / "input.py").exists() and (base_dir / "output.py").exists():
        from input import DataReader  # type: ignore
        from output import DataOutput  # type: ignore
    else:
        _ensure_sys_path(Path("/root/测试集接口"))
        from input import DataReader  # type: ignore
        from output import DataOutput  # type: ignore

    repo = Path(args.repo)

    cfg_path = Path(args.config_json) if args.config_json else (base_dir / "best_cfg.json")
    if not cfg_path.exists():
        # fallback to workspace default if available
        fallback = Path("/tmp/项目/reports/notrim_m2/best_cfg_epoch79.json")
        cfg_path = fallback if fallback.exists() else cfg_path
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    mapping_path = base_dir / "mapping.txt"
    if not mapping_path.exists():
        mapping_path = repo / "data" / args.dataset / "mapping.txt"
    idx_to_cat, labels = _load_mapping(mapping_path)
    cat_to_idx = {v: k for k, v in idx_to_cat.items()}

    import torch
    from mstcn2_model_min import MS_TCN2

    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    feats_dim = 58
    hp = {"num_layers_PG": 12, "num_layers_R": 12, "num_R": 5, "num_f_maps": 256}

    model_paths = []
    if args.model_paths.strip():
        for s in str(args.model_paths).split(","):
            s = s.strip()
            if not s:
                continue
            model_paths.append(Path(s))
    else:
        # Default: if packaged models exist, use all of them for ensembling.
        packaged = sorted((base_dir / "models").glob("*.model"))
        if packaged:
            model_paths = packaged
        else:
            model_dir = repo / "models" / args.dataset / f"split_{args.split}"
            model_paths = [model_dir / f"epoch-{int(args.epoch)}.model"]

    for mp in model_paths:
        if not mp.exists():
            raise SystemExit(f"missing model: {mp}")

    n_classes = int(len(labels))
    nets = []
    for mp in model_paths:
        net = MS_TCN2(
            int(hp["num_layers_PG"]),
            int(hp["num_layers_R"]),
            int(hp["num_R"]),
            int(hp["num_f_maps"]),
            int(feats_dim),
            n_classes,
        ).to(dev)
        net.load_state_dict(torch.load(str(mp), map_location="cpu"))
        net.eval()
        nets.append(net)

    # Read test data
    # Keep organizer-provided DataReader in the executable (per requirements),
    # but use robust reading to avoid silent skips due to encoding issues.
    _ = DataReader(args.test_data_dir)
    data = _read_test_dir_robust(Path(args.test_data_dir))  # {file_id: text}
    if not data:
        raise SystemExit(f"no test txt read from {args.test_data_dir}")

    enter_th, exit_th = _build_enter_exit_thresholds(
        idx_to_cat=idx_to_cat,
        enter_default=float(cfg.get("prob_th", 0.5)),
        class_enter=dict(cfg.get("class_enter", {})),
        hyst_delta=float(cfg.get("hyst_delta", 0.05)),
        exit_floor=float(cfg.get("exit_floor", 0.3)),
    )
    min_dur = cfg.get("class_min_dur_ms", int(cfg.get("min_dur_ms", 3000)))
    seg_score = cfg.get("class_seg_score_th", float(cfg.get("seg_score_th", 0.0)))

    results = []
    for uid in sorted(data.keys()):
        built = _build_windows_for_user(
            data[uid],
            window_ms=int(args.window_ms),
            stride_ms=int(args.stride_ms),
            min_points=int(args.min_points),
        )
        if built is None:
            continue
        starts, feats = built
        feats = _normalize_feats_user(feats, clip=10.0)
        x = torch.from_numpy(feats).unsqueeze(0).to(dev)  # (1, D, T)
        with torch.no_grad():
            logits_sum = None
            for net in nets:
                out = net(x)  # (S, 1, C, T)
                logits = out[-1, 0]  # (C, T)
                logits_sum = logits if logits_sum is None else (logits_sum + logits)
            logits_mean = logits_sum / float(len(nets))
            prob = torch.softmax(logits_mean, dim=0).transpose(0, 1).contiguous().cpu().numpy()  # (T, C)

        prob = _apply_temperature(prob, float(cfg.get("temp", 1.0)))
        prob = _smooth_prob(prob, int(cfg.get("smooth", 1)))
        segs = _segments_from_prob_hyst(
            starts=starts,
            prob=prob,
            idx_to_cat=idx_to_cat,
            cat_to_idx=cat_to_idx,
            window_ms=int(args.window_ms),
            gap_ms=int(cfg.get("gap_ms", 1280)),
            margin=float(cfg.get("margin", 0.0)),
            enter_th=enter_th,
            exit_th=exit_th,
            min_dur_ms=min_dur,
            seg_score_th=seg_score,
        )
        for cat, st, ed, _sc in segs:
            results.append([str(uid), str(cat), int(st), int(ed)])

    if not results:
        raise SystemExit("no segments predicted (check thresholds/config)")

    DataOutput(results, output_file=str(args.out_xlsx)).save_submission()


if __name__ == "__main__":
    main()

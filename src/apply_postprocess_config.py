#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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
    ap = argparse.ArgumentParser(description="Apply tuned postprocess config to cache and write pred csv + stats.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--config-json", required=True)
    ap.add_argument("--labels-json", required=True, help="cache meta.json containing labels")
    ap.add_argument("--user-list", required=True, help="users to include (stems)")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-stats", default="")
    ap.add_argument("--window-ms", type=int, default=2560)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    meta = json.loads(Path(args.labels_json).read_text(encoding="utf-8"))
    labels = meta.get("labels")
    if not labels:
        raise SystemExit("labels missing in labels-json meta")
    idx_to_cat = {int(i): str(labels[int(i)]) for i in range(len(labels))}
    cat_to_idx = {v: k for k, v in idx_to_cat.items()}

    users = [x.strip() for x in Path(args.user_list).read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
    cache_dir = Path(args.cache_dir)

    enter_th, exit_th = _build_enter_exit_thresholds(
        idx_to_cat=idx_to_cat,
        enter_default=float(cfg.get("prob_th", 0.5)),
        class_enter=dict(cfg.get("class_enter", {})),
        hyst_delta=float(cfg.get("hyst_delta", 0.05)),
        exit_floor=float(cfg.get("exit_floor", 0.3)),
    )

    min_dur = cfg.get("class_min_dur_ms", int(cfg.get("min_dur_ms", 3000)))
    seg_score = cfg.get("class_seg_score_th", float(cfg.get("seg_score_th", 0.0)))

    rows = ["user_id,category,start,end"]
    stats = {"n_users": 0, "n_segments": 0, "by_category": {}}
    for uid in users:
        p = cache_dir / f"{uid}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        starts = z["starts"].astype(np.int64, copy=False)
        prob = z["prob"].astype(np.float32, copy=False)
        if prob.ndim != 2 or starts.shape[0] != prob.shape[0]:
            continue
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
        stats["n_users"] += 1
        for cat, st, ed, sc in segs:
            rows.append(f"{uid},{cat},{int(st)},{int(ed)}")
            stats["n_segments"] += 1
            bc = stats["by_category"].setdefault(cat, {"n": 0, "dur_ms_sum": 0})
            bc["n"] += 1
            bc["dur_ms_sum"] += int(ed) - int(st)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    if args.out_stats:
        for cat, v in stats["by_category"].items():
            v["dur_ms_mean"] = (v["dur_ms_sum"] / v["n"]) if v["n"] else 0.0
        Path(args.out_stats).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


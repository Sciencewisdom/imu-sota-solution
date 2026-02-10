#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import multiprocessing as mp

if "/root" not in sys.path:
    sys.path.insert(0, "/root")

_MP_G = {}


def _mp_init_worker(caches, gt, cats, users, idx_to_cat, cat_to_idx, window_ms: int):
    _MP_G["caches"] = caches
    _MP_G["gt"] = gt
    _MP_G["cats"] = cats
    _MP_G["users"] = users
    _MP_G["idx_to_cat"] = idx_to_cat
    _MP_G["cat_to_idx"] = cat_to_idx
    _MP_G["window_ms"] = int(window_ms)


def _mp_eval_cfg(cfg):
    idx_to_cat = _MP_G["idx_to_cat"]
    cat_to_idx = _MP_G["cat_to_idx"]
    enter_th, exit_th = _build_enter_exit_thresholds(
        idx_to_cat,
        float(cfg["prob_th"]),
        dict(cfg.get("class_enter", {})),
        float(cfg["hyst_delta"]),
        float(cfg["exit_floor"]),
    )
    min_dur = cfg.get("class_min_dur_ms", int(cfg["min_dur_ms"]))
    seg_score = cfg.get("class_seg_score_th", float(cfg["seg_score_th"]))
    pred = {}
    for uid, (starts, prob) in _MP_G["caches"].items():
        p2 = _apply_temperature(prob, float(cfg["temp"]))
        p2 = _smooth_prob(p2, int(cfg["smooth"]))
        segs = _segments_from_prob_hyst(
            starts,
            p2,
            idx_to_cat,
            cat_to_idx,
            window_ms=int(_MP_G["window_ms"]),
            gap_ms=int(cfg["gap_ms"]),
            margin=float(cfg["margin"]),
            enter_th=enter_th,
            exit_th=exit_th,
            min_dur_ms=min_dur,
            seg_score_th=seg_score,
        )
        pred[uid] = [(st, ed, cat) for cat, st, ed, sc in segs]
    macro_f1, _ = _eval_segmental_f1_macro(_MP_G["gt"], pred, _MP_G["users"], _MP_G["cats"], 0.5)
    return float(macro_f1), cfg


def _load_segments_csv(p: Path):
    import csv
    from collections import defaultdict

    segs = defaultdict(list)
    cats = set()
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            uid = (r.get("user_id") or "").strip()
            cat = (r.get("category") or "").strip()
            if not uid or not cat:
                continue
            try:
                st = int(r.get("start", ""))
                ed = int(r.get("end", ""))
            except Exception:
                continue
            if ed <= st:
                continue
            segs[uid].append((st, ed, cat))
            cats.add(cat)
    for uid in segs:
        segs[uid].sort(key=lambda x: (x[2], x[0], x[1]))
    return dict(segs), sorted(cats)


def _iou(a0, a1, b0, b1):
    inter = max(0, min(a1, b1) - max(a0, b0))
    if inter <= 0:
        return 0.0
    uni = (a1 - a0) + (b1 - b0) - inter
    return inter / uni if uni > 0 else 0.0


def _match_one_class(gt, pr, iou_th):
    used = [False] * len(pr)
    tp = 0
    for g0, g1 in gt:
        best = -1
        best_i = 0.0
        for j, (p0, p1) in enumerate(pr):
            if used[j]:
                continue
            v = _iou(g0, g1, p0, p1)
            if v > best_i:
                best_i = v
                best = j
        if best >= 0 and best_i >= iou_th:
            used[best] = True
            tp += 1
    fp = used.count(False)
    fn = len(gt) - tp
    return tp, fp, fn


def _eval_segmental_f1_macro(gt_segs, pr_segs, users, cats, iou_th):
    per_cat = {}
    f1s = []
    for cat in cats:
        tp = fp = fn = 0
        for u in users:
            g = [(a, b) for a, b, c in gt_segs.get(u, []) if c == cat]
            p = [(a, b) for a, b, c in pr_segs.get(u, []) if c == cat]
            tpi, fpi, fni = _match_one_class(g, p, iou_th)
            tp += tpi
            fp += fpi
            fn += fni
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_cat[cat] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0, per_cat


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


def _apply_temperature(prob: np.ndarray, temp: float) -> np.ndarray:
    # Temperature scaling on softmax probabilities without logits:
    # if prob = softmax(z), then softmax(z / T) == normalize(prob ** (1/T)).
    t = float(temp)
    if not np.isfinite(t) or t <= 0:
        return prob
    if abs(t - 1.0) < 1e-6:
        return prob
    p = np.clip(prob, 1e-8, 1.0).astype(np.float32, copy=False)
    p = p ** (1.0 / t)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return p.astype(np.float32, copy=False)


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
    ap = argparse.ArgumentParser(description="Tune segment postprocess thresholds on cached probs (npz).")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--label-file", default="/root/csv_files/赛题2金标-训练集_Sheet1.csv")
    ap.add_argument("--id-alias-file", default="/root/csv_files/赛题2异常数据说明-训练集_训练集.csv")
    ap.add_argument("--user-list", required=True, help="Users to evaluate (one per line, stem only).")
    ap.add_argument("--labels-json", default="", help="Optional meta.json with labels list. If absent, infer from prob dim.")
    ap.add_argument("--window-ms", type=int, default=2560)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default="best_config.json")
    ap.add_argument("--per-class", action="store_true", help="Enable per-class thresholds (prob_th/min_dur/seg_score).")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel workers for tuning (CPU).")
    args = ap.parse_args()

    import train_tsv_baseline as fe

    alias = fe.load_id_aliases(Path(args.id_alias_file))
    gt, cats = _load_segments_csv(Path(args.label_file))
    gt = fe.apply_aliases(gt, alias)

    users = [x.strip() for x in Path(args.user_list).read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]

    cache_dir = Path(args.cache_dir)
    # load all caches
    caches = {}
    first = None
    for uid in users:
        p = cache_dir / f"{uid}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        starts = z["starts"].astype(np.int64, copy=False)
        prob = z["prob"].astype(np.float32, copy=False)
        if prob.ndim != 2 or starts.shape[0] != prob.shape[0]:
            continue
        caches[uid] = (starts, prob)
        if first is None:
            first = prob
    if not caches:
        raise SystemExit("No caches found for users.")

    if args.labels_json:
        meta = json.loads(Path(args.labels_json).read_text(encoding="utf-8"))
        labels = meta["labels"]
    else:
        # Assume background + cats order is already consistent; just build placeholder names.
        labels = ["background"] + list(cats)

    idx_to_cat = {i: labels[i] for i in range(len(labels))}
    cat_to_idx = {c: i for i, c in idx_to_cat.items()}

    rng = np.random.default_rng(int(args.seed))
    best = {"macro_f1": -1.0}

    prob_th_choices = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    smooth_choices = [1, 3, 5, 7, 9, 13]
    min_dur_choices = [2560, 3000, 4000, 6000, 8000]
    gap_choices = [0, 640, 1280, 1920, 2560]
    margin_choices = [0.0, 0.02, 0.05, 0.1]
    seg_score_choices = [0.0, 0.2, 0.4, 0.6, 0.75]
    hyst_delta_choices = [0.0, 0.05, 0.1, 0.15]
    exit_floor_choices = [0.1, 0.2, 0.25, 0.3]
    temp_choices = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2]

    def sample_cfg(_rng: np.random.Generator):
        cfg = dict(
            prob_th=float(_rng.choice(prob_th_choices)),
            smooth=int(_rng.choice(smooth_choices)),
            min_dur_ms=int(_rng.choice(min_dur_choices)),
            gap_ms=int(_rng.choice(gap_choices)),
            margin=float(_rng.choice(margin_choices)),
            seg_score_th=float(_rng.choice(seg_score_choices)),
            hyst_delta=float(_rng.choice(hyst_delta_choices)),
            exit_floor=float(_rng.choice(exit_floor_choices)),
            temp=float(_rng.choice(temp_choices)),
        )
        if bool(args.per_class):
            class_enter = {}
            class_min_dur = {}
            class_seg_score = {}
            for cat in cats:
                class_enter[cat] = float(_rng.choice(prob_th_choices))
                class_min_dur[cat] = int(_rng.choice(min_dur_choices))
                class_seg_score[cat] = float(_rng.choice(seg_score_choices))
            cfg["class_enter"] = class_enter
            cfg["class_min_dur_ms"] = class_min_dur
            cfg["class_seg_score_th"] = class_seg_score
        return cfg

    def _eval_cfg_local(cfg):
        enter_th, exit_th = _build_enter_exit_thresholds(
            idx_to_cat,
            float(cfg["prob_th"]),
            dict(cfg.get("class_enter", {})),
            float(cfg["hyst_delta"]),
            float(cfg["exit_floor"]),
        )
        min_dur = cfg.get("class_min_dur_ms", int(cfg["min_dur_ms"]))
        seg_score = cfg.get("class_seg_score_th", float(cfg["seg_score_th"]))
        pred = {}
        for uid, (starts, prob) in caches.items():
            p2 = _apply_temperature(prob, float(cfg["temp"]))
            p2 = _smooth_prob(p2, int(cfg["smooth"]))
            segs = _segments_from_prob_hyst(
                starts,
                p2,
                idx_to_cat,
                cat_to_idx,
                window_ms=int(args.window_ms),
                gap_ms=int(cfg["gap_ms"]),
                margin=float(cfg["margin"]),
                enter_th=enter_th,
                exit_th=exit_th,
                min_dur_ms=min_dur,
                seg_score_th=seg_score,
            )
            pred[uid] = [(st, ed, cat) for cat, st, ed, sc in segs]
        macro_f1, _ = _eval_segmental_f1_macro(gt, pred, users, cats, 0.5)
        return float(macro_f1), cfg

    iters = int(args.iters)
    jobs = int(args.jobs)
    if jobs <= 1:
        for _it in range(iters):
            cfg = sample_cfg(rng)
            macro_f1, _cfg = _eval_cfg_local(cfg)
            if macro_f1 > float(best["macro_f1"]):
                best = dict(_cfg)
                best["macro_f1"] = float(macro_f1)
    else:
        # Fork is preferred here for speed (small caches, read-only, copy-on-write).
        ctx = mp.get_context("fork")
        pool = ctx.Pool(
            processes=jobs,
            initializer=_mp_init_worker,
            initargs=(caches, gt, cats, users, idx_to_cat, cat_to_idx, int(args.window_ms)),
        )
        # Sample configs on the main process (deterministic seed).
        cfgs = [sample_cfg(rng) for _ in range(iters)]
        for macro_f1, cfg in pool.imap_unordered(_mp_eval_cfg, cfgs, chunksize=8):
            if macro_f1 > float(best["macro_f1"]):
                best = dict(cfg)
                best["macro_f1"] = float(macro_f1)
        pool.close()
        pool.join()

    # Compute per-category breakdown for the best config.
    enter_th, exit_th = _build_enter_exit_thresholds(
        idx_to_cat, float(best.get("prob_th", 0.5)), dict(best.get("class_enter", {})), float(best.get("hyst_delta", 0.05)), float(best.get("exit_floor", 0.3))
    )
    min_dur = best.get("class_min_dur_ms", int(best.get("min_dur_ms", 3000)))
    seg_score = best.get("class_seg_score_th", float(best.get("seg_score_th", 0.0)))
    pred = {}
    for uid, (starts, prob) in caches.items():
        p2 = _apply_temperature(prob, float(best.get("temp", 1.0)))
        p2 = _smooth_prob(p2, int(best.get("smooth", 1)))
        segs = _segments_from_prob_hyst(
            starts,
            p2,
            idx_to_cat,
            cat_to_idx,
            window_ms=int(args.window_ms),
            gap_ms=int(best.get("gap_ms", 1280)),
            margin=float(best.get("margin", 0.0)),
            enter_th=enter_th,
            exit_th=exit_th,
            min_dur_ms=min_dur,
            seg_score_th=seg_score,
        )
        pred[uid] = [(st, ed, cat) for cat, st, ed, sc in segs]
    _macro_f1, per_cat = _eval_segmental_f1_macro(gt, pred, users, cats, 0.5)
    best["macro_f1"] = float(_macro_f1)
    best["per_category"] = per_cat

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best_macro_f1={best['macro_f1']:.6f}")
    print(f"out={out}")


if __name__ == "__main__":
    main()

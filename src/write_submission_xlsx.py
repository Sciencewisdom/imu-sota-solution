#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Convert pred csv (user_id,category,start,end) to submission.xlsx.")
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-xlsx", required=True)
    args = ap.parse_args()

    inp = Path(args.in_csv)
    if not inp.exists():
        raise SystemExit(f"missing: {inp}")

    df = pd.read_csv(inp)
    need = ["user_id", "category", "start", "end"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"missing column {c} in {inp} (has {list(df.columns)})")

    df = df[need].copy()
    df["user_id"] = df["user_id"].astype(str)
    df["category"] = df["category"].astype(str)
    # enforce int64
    df["start"] = df["start"].astype("int64")
    df["end"] = df["end"].astype("int64")

    # Basic sanity checks
    bad = (df["end"] <= df["start"]).sum()
    if bad:
        raise SystemExit(f"found {bad} rows with end<=start in {inp}")

    out = Path(args.out_xlsx)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out, index=False, engine="openpyxl")
    print(f"wrote={out} rows={len(df)}")


if __name__ == "__main__":
    main()


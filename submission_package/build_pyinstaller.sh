#!/usr/bin/env bash
set -euo pipefail

# Build a standalone executable with pyinstaller.
# Note: bundling torch will produce a large binary/folder.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

OUTDIR="${OUTDIR:-dist_submit}"
NAME="${NAME:-submit_runner}"

rm -rf build dist "$OUTDIR" "${NAME}.spec" || true

python3 -m PyInstaller \
  --name "$NAME" \
  --clean \
  --noconfirm \
  --onedir \
  --distpath "$OUTDIR" \
  --add-data "best_cfg.json:." \
  --add-data "mapping.txt:." \
  --add-data "input.py:." \
  --add-data "output.py:." \
  --add-data "models:models" \
  --collect-all torch \
  --collect-all numpy \
  --collect-all pandas \
  --collect-all openpyxl \
  run.py

echo "built: $OUTDIR/$NAME"


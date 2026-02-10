#!/bin/bash

set -euo pipefail

DATASET="${1}"
SPLIT="${2}"

if [[ "${DATASET}" == "imu_sota" ]]; then
  python3 main.py --action=train --dataset="${DATASET}" --split="${SPLIT}" \
                  --features_dim=58 \
                  --bz=16 \
                  --lr=0.0005 \
                  --num_f_maps=256 \
                  --num_epochs=80 \
                  --num_layers_PG=12 \
                  --num_layers_R=12 \
                  --num_R=5 \
                  --preload \
                  --best_metric=f1_50 \
                  --save_best=1 --save_final=1 --val_every=1 \
                  --save_every=5 --save_opt=0
elif [[ "${DATASET}" == "imu_sota_xgb_norm" ]] || [[ "${DATASET}" == "imu_sota_xgb" ]]; then
  python3 main.py --action=train --dataset="${DATASET}" --split="${SPLIT}" \
                  --features_dim=64 \
                  --bz=16 \
                  --lr=0.0005 \
                  --num_f_maps=256 \
                  --num_epochs=80 \
                  --num_layers_PG=12 \
                  --num_layers_R=12 \
                  --num_R=5 \
                  --preload \
                  --best_metric=f1_50 \
                  --save_best=1 --save_final=1 --val_every=1 \
                  --save_every=5 --save_opt=0
else
  python3 main.py --action=train --dataset="${DATASET}" --split="${SPLIT}" \
                  --num_epochs=100 \
                  --num_layers_PG=11 \
                  --num_layers_R=10 \
                  --num_R=3
fi

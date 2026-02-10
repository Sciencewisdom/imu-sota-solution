#!/bin/bash
set -euo pipefail

DATASET="${1}"
SPLIT="${2}"
EPOCH="${3}"

if [[ "${DATASET}" == "imu_sota" ]] || [[ "${DATASET}" == "imu_sota_notrim" ]]; then
python3 main.py --action=predict --dataset="${DATASET}" --split="${SPLIT}" --num_epochs="${EPOCH}" \
               --features_dim=58 \
               --num_f_maps=256 \
               --num_layers_PG=12 \
               --num_layers_R=12 \
               --num_R=5
elif [[ "${DATASET}" == "imu_sota_xgb_norm" ]] || [[ "${DATASET}" == "imu_sota_xgb" ]]; then
python3 main.py --action=predict --dataset="${DATASET}" --split="${SPLIT}" --num_epochs="${EPOCH}" \
               --features_dim=64 \
               --num_f_maps=256 \
               --num_layers_PG=12 \
               --num_layers_R=12 \
               --num_R=5
else
python3 main.py --action=predict --dataset="${DATASET}" --split="${SPLIT}" --num_epochs="${EPOCH}" \
               --num_layers_PG=11 \
               --num_layers_R=10 \
               --num_R=3
fi

python3 eval.py --dataset="${DATASET}" --split="${SPLIT}"

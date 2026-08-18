#!/usr/bin/env bash
set -euo pipefail

repo=/home/dgxsparktd/activsg-gpu-scopf-spark
mkdir -p \
  "$repo/results/cupy-cache" \
  "$repo/results/cuda-cache"

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  --env GIT_CONFIG_COUNT=1 \
  --env GIT_CONFIG_KEY_0=safe.directory \
  --env GIT_CONFIG_VALUE_0=/workspace \
  --env CUPY_CACHE_DIR=/workspace/results/cupy-cache \
  --env CUDA_CACHE_PATH=/workspace/results/cuda-cache \
  --volume "$repo:/workspace:ro" \
  --volume "$repo/results:/workspace/results:rw" \
  activsg-gpu-scopf-spark:benchmark-10k-v3-diagnostic \
  solve \
  --config /workspace/configs/activsg10k-v3.json \
  --platform dgx_spark \
  --output /workspace/results/activsg10k-v3-dgx-spark-authorized-diagnostic.json

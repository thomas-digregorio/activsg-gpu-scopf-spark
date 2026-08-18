#!/usr/bin/env bash
set -euo pipefail

repo=/home/dgxsparktd/activsg-gpu-scopf-spark
mkdir -p "$repo/results"

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  --env GIT_CONFIG_COUNT=1 \
  --env GIT_CONFIG_KEY_0=safe.directory \
  --env GIT_CONFIG_VALUE_0=/workspace \
  --volume "$repo:/workspace:ro" \
  --volume "$repo/results:/workspace/results:rw" \
  activsg-gpu-scopf-spark:benchmark-10k-v1 \
  benchmark \
  --config /workspace/configs/activsg10k.json \
  --platform dgx_spark \
  --laptop-result /workspace/results/activsg10k-laptop-cpu-official.json \
  --output /workspace/results/activsg10k-dgx-spark-official.json

#!/usr/bin/env bash
set -euo pipefail

repo=/home/dgxsparktd/activsg-gpu-scopf-spark
tag=experiment-2000-gpu-gap-v2
image=activsg-gpu-scopf-spark:experiment-2000-gpu-gap-v2
output=activsg2000-gpu-gap-v2-1e-3-dgx-spark.json
cache_root="$repo/results/cache/activsg2000-gpu-gap-v2"

cd "$repo"
head_commit=$(git rev-parse HEAD)
tag_commit=$(git rev-list -n 1 "$tag")
if [[ "$head_commit" != "$tag_commit" ]]; then
  echo "Refusing run: HEAD $head_commit does not equal $tag $tag_commit" >&2
  exit 1
fi
if [[ -e "$repo/results/experiments/$output" ]]; then
  echo "Refusing run: output already exists: $repo/results/experiments/$output" >&2
  exit 1
fi
if [[ -e "$repo/results/experiments/activsg2000-gpu-gap-sensitivity-v2-run-registry.json" ]]; then
  echo "Refusing run: GPU v2 suite registry already exists" >&2
  exit 1
fi
if [[ -e "$cache_root" ]]; then
  echo "Refusing run: suite-specific cache already exists: $cache_root" >&2
  exit 1
fi

mkdir -p \
  "$repo/results/experiments" \
  "$repo/results/checkpoints" \
  "$repo/results/diagnostics" \
  "$cache_root/cupy" \
  "$cache_root/cuda" \
  "$cache_root/xdg"

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  --env GIT_CONFIG_COUNT=1 \
  --env GIT_CONFIG_KEY_0=safe.directory \
  --env GIT_CONFIG_VALUE_0=/workspace \
  --env CUPY_CACHE_DIR=/workspace/results/cache/activsg2000-gpu-gap-v2/cupy \
  --env CUDA_CACHE_PATH=/workspace/results/cache/activsg2000-gpu-gap-v2/cuda \
  --env XDG_CACHE_HOME=/workspace/results/cache/activsg2000-gpu-gap-v2/xdg \
  --volume "$repo:/workspace:ro" \
  --volume "$repo/results:/workspace/results:rw" \
  "$image" \
  gap-experiment \
  --config /workspace/configs/activsg2000-gpu-gap-1e-3-v2.json \
  --output "/workspace/results/experiments/$output"

#!/usr/bin/env bash
set -euo pipefail

repo=/home/dgxsparktd/activsg-gpu-scopf-spark
tag=experiment-2000-gpu-lagrangian-v8
image=activsg-gpu-scopf-spark:experiment-2000-gpu-lagrangian-v8
suite=activsg2000-gpu-lagrangian-v8
output="$suite-dgx-spark.json"
console_log="$repo/results/experiments/$suite-launcher.log"
cache_root="$repo/results/cache/$suite"

cd "$repo"
head_commit=$(git rev-parse HEAD)
tag_commit=$(git rev-list -n 1 "$tag")
if [[ "$head_commit" != "$tag_commit" ]]; then
  echo "Refusing run: HEAD $head_commit does not equal tag $tag_commit" >&2
  exit 1
fi
for path in \
  "$repo/results/experiments/$output" \
  "$repo/results/experiments/$suite-run-registry.json" \
  "$repo/results/checkpoints/$suite-dgx_spark.json" \
  "$repo/results/diagnostics/$suite-worker-console.log" \
  "$console_log" \
  "$cache_root"; do
  if [[ -e "$path" ]]; then
    echo "Refusing run: one-shot path already exists: $path" >&2
    exit 1
  fi
done

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
  --env CUPY_CACHE_DIR="/workspace/results/cache/$suite/cupy" \
  --env CUDA_CACHE_PATH="/workspace/results/cache/$suite/cuda" \
  --env XDG_CACHE_HOME="/workspace/results/cache/$suite/xdg" \
  --volume "$repo:/workspace:ro" \
  --volume "$repo/results:/workspace/results:rw" \
  "$image" \
  gpu-lagrangian-experiment \
  --config /workspace/configs/activsg2000-gpu-lagrangian-v8.json \
  --output "/workspace/results/experiments/$output" \
  2>&1 | tee "$console_log"

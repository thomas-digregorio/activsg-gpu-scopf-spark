#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <1e-3|1e-4|1e-5|1e-6|1e-7>" >&2
  exit 2
fi

label=$1
case "$label" in
  1e-3|1e-4|1e-5|1e-6|1e-7) ;;
  *)
    echo "Unregistered gap label: $label" >&2
    exit 2
    ;;
esac

repo=/home/dgxsparktd/activsg-gpu-scopf-spark
tag=experiment-500-gpu-gap-v1
image=activsg-gpu-scopf-spark:experiment-500-gpu-gap-v1
output="activsg500-gpu-gap-v1-${label}-dgx-spark.json"

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

mkdir -p \
  "$repo/results/experiments" \
  "$repo/results/checkpoints" \
  "$repo/results/diagnostics" \
  "$repo/results/cache/cupy" \
  "$repo/results/cache/cuda" \
  "$repo/results/cache/xdg"

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  --env GIT_CONFIG_COUNT=1 \
  --env GIT_CONFIG_KEY_0=safe.directory \
  --env GIT_CONFIG_VALUE_0=/workspace \
  --env CUPY_CACHE_DIR=/workspace/results/cache/cupy \
  --env CUDA_CACHE_PATH=/workspace/results/cache/cuda \
  --env XDG_CACHE_HOME=/workspace/results/cache/xdg \
  --volume "$repo:/workspace:ro" \
  --volume "$repo/results:/workspace/results:rw" \
  "$image" \
  gap-experiment \
  --config "/workspace/configs/activsg500-gpu-gap-${label}.json" \
  --output "/workspace/results/experiments/$output"

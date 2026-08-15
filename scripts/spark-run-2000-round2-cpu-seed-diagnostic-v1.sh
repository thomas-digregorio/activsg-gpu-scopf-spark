#!/usr/bin/env bash
set -euo pipefail

repo=/home/dgxsparktd/activsg-gpu-scopf-spark
tag=diagnostic-2000-gpu-round2-cpu-seed-v1
image=activsg-gpu-scopf-spark:diagnostic-2000-gpu-round2-cpu-seed-v1
diagnostic_id=activsg2000-gpu-round2-cpu-seed-diagnostic-v1
output="$repo/results/diagnostics/$diagnostic_id-dgx-spark.json"
console_log="$repo/results/diagnostics/$diagnostic_id-cuopt-console.log"
registry="$repo/results/diagnostics/$diagnostic_id-run-registry.json"
checkpoint="$repo/results/checkpoints/$diagnostic_id-dgx_spark.json"
cache_root="$repo/results/cache/$diagnostic_id"
cpu_input="$repo/results/diagnostic-inputs/activsg2000-gap-v1-1e-3-laptop.json"
gpu_input="$repo/results/diagnostic-inputs/activsg2000-gpu-gap-v3-1e-3-dgx-spark.json"
cpu_sha=5573425a8e625c0c964b2c61d33ca80666de74a350e9431a96e5ce9b90e02e3f
gpu_sha=9b9e7d3b0dfdad0faeb740f639307844920c18eb85c07ac9705f8b84e034366c

cd "$repo"
head_commit=$(git rev-parse HEAD)
tag_commit=$(git rev-list -n 1 "$tag")
if [[ "$head_commit" != "$tag_commit" ]]; then
  echo "Refusing run: HEAD $head_commit does not equal $tag $tag_commit" >&2
  exit 1
fi
if [[ $(sha256sum "$cpu_input" | awk '{print $1}') != "$cpu_sha" ]]; then
  echo "Refusing run: secure CPU result hash mismatch" >&2
  exit 1
fi
if [[ $(sha256sum "$gpu_input" | awk '{print $1}') != "$gpu_sha" ]]; then
  echo "Refusing run: GPU v3 result hash mismatch" >&2
  exit 1
fi
for path in "$output" "$console_log" "$registry" "$checkpoint" "$cache_root"; do
  if [[ -e "$path" ]]; then
    echo "Refusing run: one-shot diagnostic path already exists: $path" >&2
    exit 1
  fi
done

mkdir -p \
  "$repo/results/diagnostics" \
  "$repo/results/checkpoints" \
  "$repo/results/diagnostic-inputs" \
  "$cache_root/cupy" \
  "$cache_root/cuda" \
  "$cache_root/xdg"

docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  --env GIT_CONFIG_COUNT=1 \
  --env GIT_CONFIG_KEY_0=safe.directory \
  --env GIT_CONFIG_VALUE_0=/workspace \
  --env CUPY_CACHE_DIR="/workspace/results/cache/$diagnostic_id/cupy" \
  --env CUDA_CACHE_PATH="/workspace/results/cache/$diagnostic_id/cuda" \
  --env XDG_CACHE_HOME="/workspace/results/cache/$diagnostic_id/xdg" \
  --volume "$repo:/workspace:ro" \
  --volume "$repo/results:/workspace/results:rw" \
  "$image" \
  seeded-round2-diagnostic \
  --config /workspace/configs/activsg2000-gpu-round2-cpu-seed-diagnostic-v1.json \
  --output "/workspace/results/diagnostics/$diagnostic_id-dgx-spark.json" \
  2>&1 | tee "$console_log"

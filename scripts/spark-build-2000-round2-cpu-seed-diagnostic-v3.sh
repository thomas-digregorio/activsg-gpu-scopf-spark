#!/usr/bin/env bash
set -euo pipefail

repo=/home/dgxsparktd/activsg-gpu-scopf-spark
tag=diagnostic-2000-gpu-round2-cpu-seed-v3
image=activsg-gpu-scopf-spark:diagnostic-2000-gpu-round2-cpu-seed-v3

cd "$repo"
head_commit=$(git rev-parse HEAD)
tag_commit=$(git rev-list -n 1 "$tag")
if [[ "$head_commit" != "$tag_commit" ]]; then
  echo "Refusing build: HEAD $head_commit does not equal tag $tag_commit" >&2
  exit 1
fi

docker build \
  --build-arg "SOURCE_COMMIT=$head_commit" \
  --tag "$image" \
  --file Dockerfile.spark \
  .

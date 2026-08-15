#!/usr/bin/env bash
set -euo pipefail

repo=/home/dgxsparktd/activsg-gpu-scopf-spark
cd "$repo"
commit=$(git rev-parse HEAD)
docker build \
  --build-arg "SOURCE_COMMIT=$commit" \
  --tag activsg-gpu-scopf-spark:benchmark-10k-v1 \
  --file Dockerfile.spark \
  .

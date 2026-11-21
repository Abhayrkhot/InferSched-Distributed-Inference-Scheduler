#!/bin/bash
# Converts the CUDA build image to Apptainer and compiles InferSched with the
# llama.cpp backend (plus llama-server) for A100 and H200 (sm_80, sm_90). CPU-only job.
#   cd /scratch/$USER/infersched/src && sbatch scripts/explorer/build.sbatch
# Expects /scratch/$USER/infersched/infersched-gpu.tar.gz (docker save of
# docker/gpu.Dockerfile, gzipped).
#SBATCH --job-name=infersched-build
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/infersched/logs/build-%j.out
set -euo pipefail
ROOT=/scratch/$USER/infersched
SIF=$ROOT/infersched-gpu.sif
if [ ! -f "$SIF" ]; then
  # Unpack on node-local disk; unpacking on /scratch (NFS) is very slow.
  export APPTAINER_TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/infersched-apptainer-XXXXXX")
  trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT
  gunzip -c "$ROOT/infersched-gpu.tar.gz" > "$APPTAINER_TMPDIR/image.tar"
  apptainer build "$APPTAINER_TMPDIR/image.sif" "docker-archive://$APPTAINER_TMPDIR/image.tar"
  cp "$APPTAINER_TMPDIR/image.sif" "$SIF.partial" && mv "$SIF.partial" "$SIF"
fi
echo "== image $(du -h "$SIF" | cut -f1) on $(hostname)"
cd "$ROOT/src"
# FetchContent downloads googletest and the pinned llama.cpp tarball through the
# compute node's HTTP proxy, so the environment is passed through.
apptainer exec --bind "$ROOT" "$SIF" bash -c '
  set -euo pipefail
  cmake -S . -B build/llama-a100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DINFERSCHED_WITH_LLAMA=ON -DINFERSCHED_LLAMA_CUDA=ON -DINFERSCHED_LLAMA_TOOLS=ON \
    "-DCMAKE_CUDA_ARCHITECTURES=80;90" > build-configure.log
  cmake --build build/llama-a100 -j "$SLURM_CPUS_PER_TASK" 2>&1 | grep -E "error|FAILED" || true
  cmake --build build/llama-a100 -j "$SLURM_CPUS_PER_TASK" --target llama-server llama-batched-bench \
    2>&1 | grep -E "error|FAILED" || true
  ls -la build/llama-a100/cpp/bench/infersched_real_serving_bench build/llama-a100/cpp/infersched_engine \
    build/llama-a100/bin/llama-server build/llama-a100/cpp/tests/infersched_llama_test'
echo "== built $(date -Is)"

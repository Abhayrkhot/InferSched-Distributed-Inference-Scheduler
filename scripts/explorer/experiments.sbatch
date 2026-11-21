#!/bin/bash
# Runs the P10 real-model experiments on one A100 (scripts/exp_real_serving.py).
#   cd /scratch/$USER/infersched/src && sbatch scripts/explorer/experiments.sbatch <phases> [driver flags]
# Example: sbatch scripts/explorer/experiments.sbatch calibrate batching \
#            --gguf gguf/Qwen3-8B-F16.gguf --hf-model hf/Qwen3-8B --kv-blocks 4096 --max-seqs 64
#SBATCH --job-name=infersched-p10
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/infersched/logs/p10-%j.out
set -euo pipefail
ROOT=/scratch/$USER/infersched
OUT=$ROOT/results/$SLURM_JOB_ID
mkdir -p "$OUT"
cd "$ROOT/src"
echo "== $(hostname) $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv,noheader | tee "$OUT/gpu.txt"
lscpu | grep -E "Model name|^CPU\(s\)" | tee -a "$OUT/gpu.txt"
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
# The real-model tests once more, on this node, before measuring anything.
apptainer exec --nv --bind "$ROOT" --env INFERSCHED_TEST_MODEL="$ROOT/models/gguf/stories260K.gguf" \
  "$ROOT/infersched-gpu.sif" build/llama-a100/cpp/tests/infersched_llama_test --gtest_brief=1 | tail -3
python3 scripts/exp_real_serving.py "$@" --runner apptainer \
  --gpu-image "$ROOT/infersched-gpu.sif" --vllm-image "/scratch/$USER/icp/vllm.sif" \
  --build-dir build/llama-a100 --models "$ROOT/models" --out "$OUT" \
  --vllm-memory-fraction 0.85
echo "== done $(date -Is)"

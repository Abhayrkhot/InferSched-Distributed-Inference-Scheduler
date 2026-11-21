#!/bin/bash
# Reproduces the llama.cpp b12188 CUDA-graph failure on an H200 (docs/p10-real-model-serving.md):
# steps that carry many prompts crash in ggml_cuda_graph_launch_chunk unless
# GGML_CUDA_DISABLE_GRAPHS is set; small steps work either way.
#SBATCH --job-name=infersched-diag
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/%u/infersched/logs/diag-%j.out
set -uo pipefail
ROOT=/scratch/$USER/infersched
LOGS=$ROOT/logs/diag-$SLURM_JOB_ID
mkdir -p "$LOGS"
cd "$ROOT/src"
SIF=$ROOT/infersched-gpu.sif
run() {  # run <label> <command...>
  local label=$1; shift
  echo "== $label $(date +%T)"
  apptainer exec --nv --bind "$ROOT" --bind "$ROOT/models:/models" --env CUDA_LAUNCH_BLOCKING=1 "$SIF" "$@" > "$LOGS/$label.log" 2>&1
  echo "rc=$?"
  grep -E "CUDA error|current device|GGML_ASSERT|  ggml_|error:|output_tok_s|^\|" "$LOGS/$label.log" | grep -v "Overriding HOME" | head -8 | cut -c1-300
}
BENCH="build/llama-a100/cpp/bench/infersched_real_serving_bench --model /models/gguf/Qwen3-8B-F16.gguf --max-batch 64 --kv-blocks 4096 --prefix-cache off --holders 0 --threads 8 --qps 0 --requests 128 --warmup 2 --prompt-min 128 --prompt-max 512 --short-min 16 --short-max 64 --long-min 256 --long-max 512 --long-fraction 0.2"
run static-graphs-off env GGML_CUDA_DISABLE_GRAPHS=1 $BENCH --policy static
run continuous64-graphs-off env GGML_CUDA_DISABLE_GRAPHS=1 $BENCH --policy continuous --max-prefill-seqs 64 --max-batch-tokens 65536
run continuous64-graphs-on $BENCH --policy continuous --max-prefill-seqs 64 --max-batch-tokens 65536
run continuous16-graphs-on $BENCH --policy continuous --max-prefill-seqs 16 --max-batch-tokens 4096
echo "== done $(date +%T)"

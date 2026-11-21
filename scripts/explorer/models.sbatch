#!/bin/bash
# Downloads the benchmark models and converts them to GGUF F16 for llama.cpp.
#   cd /scratch/$USER/infersched/src && sbatch scripts/explorer/models.sbatch [Qwen/Qwen3-8B ...]
# Uses the vLLM image (huggingface_hub, torch, transformers) and the convert
# script from the same pinned llama.cpp release the build uses (b12188).
#SBATCH --job-name=infersched-models
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/infersched/logs/models-%j.out
set -euo pipefail
ROOT=/scratch/$USER/infersched
VLLM_SIF=/scratch/$USER/icp/vllm.sif
LLAMA_SRC=$ROOT/llama-b12188
mkdir -p "$ROOT/models/hf" "$ROOT/models/gguf"
if [ ! -d "$LLAMA_SRC" ]; then
  curl -fsSL -o "$ROOT/llama-b12188.tar.gz" https://github.com/ggml-org/llama.cpp/archive/refs/tags/b12188.tar.gz
  echo "56b07fdceb8641325759a097d43f9f6effc61b2c7c710cb3f61e331e2b1778c6  $ROOT/llama-b12188.tar.gz" | sha256sum --check
  mkdir -p "$LLAMA_SRC" && tar xzf "$ROOT/llama-b12188.tar.gz" -C "$LLAMA_SRC" --strip-components=1
fi
export PYTHONNOUSERSITE=1 HF_HOME=$ROOT/hf-home
for repo in "${@:-Qwen/Qwen3-8B Qwen/Qwen3-0.6B}"; do
  for model in $repo; do
    name=${model##*/}
    echo "== $model $(date -Is)"
    apptainer exec --bind "$ROOT" "$VLLM_SIF" hf download "$model" --local-dir "$ROOT/models/hf/$name"
    if [ ! -f "$ROOT/models/gguf/$name-F16.gguf" ]; then
      apptainer exec --bind "$ROOT" --env PYTHONPATH="$LLAMA_SRC/gguf-py" "$VLLM_SIF" \
        python3 "$LLAMA_SRC/convert_hf_to_gguf.py" "$ROOT/models/hf/$name" --outtype f16 \
        --outfile "$ROOT/models/gguf/$name-F16.gguf.partial"
      mv "$ROOT/models/gguf/$name-F16.gguf.partial" "$ROOT/models/gguf/$name-F16.gguf"
    fi
    ls -la "$ROOT/models/gguf/$name-F16.gguf"
  done
done
curl -fsSL -o "$ROOT/models/gguf/stories260K.gguf" \
  https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf
echo "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d  $ROOT/models/gguf/stories260K.gguf" | sha256sum --check
echo "== done $(date -Is)"

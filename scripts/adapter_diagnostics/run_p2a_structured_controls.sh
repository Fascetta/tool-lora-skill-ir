#!/usr/bin/env bash
set -euo pipefail

MODE="$1"
GPU="$2"
SEED="$3"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/opt/miniconda/envs/dragdrop-vla/bin/python"

for tool in userinfoforsoundcloud playlistfordeezer; do
  output="$ROOT/experiments/stage4_p2a_structured_${MODE}_${tool}_s${SEED}"
  if [[ -f "$output/training_summary.json" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/functional_hypernet/train_free_latent_shared.py \
    --model "$ROOT/cache/models/Qwen3-0.6B" \
    --dataset-root "$ROOT/data/processed/oracle_lora_datasets_qwen3_expanded_minimal" \
    --tool-ids "$tool" \
    --output-dir "$output" \
    --device cuda \
    --max-steps 400 \
    --examples-per-tool 2 \
    --eval-interval 50 \
    --seed "$SEED" \
    --objective ce \
    --parameterization "$MODE"
done

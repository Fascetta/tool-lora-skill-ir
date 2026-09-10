#!/usr/bin/env bash
set -euo pipefail

GPU="$1"
SEED="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
MODEL="$ROOT/cache/models/Qwen3-0.6B"
DATASET="$ROOT/data/processed/oracle_lora_datasets_qwen3_expanded_minimal"
TEACHERS="$ROOT/checkpoints/oracle_loras/qwen3_0_6b"

for tool in userinfoforsoundcloud playlistfordeezer; do
  for objective in ce distill_delta_norm; do
    output="$ROOT/experiments/stage4_p2_a1r_rep_${tool}_s${SEED}_${objective}"
    if [[ -f "$output/training_summary.json" ]]; then
      continue
    fi
    args=(
      scripts/functional_hypernet/train_free_latent_shared.py
      --model "$MODEL"
      --dataset-root "$DATASET"
      --tool-ids "$tool"
      --output-dir "$output"
      --device cuda
      --max-steps 400
      --examples-per-tool 2
      --eval-interval 50
      --seed "$SEED"
      --objective "$objective"
    )
    if [[ "$objective" == "distill_delta_norm" ]]; then
      args+=(
        --lambda-regularization 1000
        --lambda-distill 0.5
        --distill-temperature 2
        --teacher-root "$TEACHERS"
      )
    fi
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "${args[@]}"
  done
done

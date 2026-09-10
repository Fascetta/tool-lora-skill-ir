#!/usr/bin/env bash
set -euo pipefail

CONDITION="$1"
GPU="$2"
SEED="$3"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
BASE="$ROOT/data/processed/oracle_lora_datasets_qwen3_expanded_minimal"
TEACHERS="$ROOT/checkpoints/oracle_loras/qwen3_0_6b"

case "$CONDITION" in
  d0_real)
    DATASET="$BASE"
    OUTPUT="$ROOT/experiments/stage4_p2a3_d0_real_s${SEED}"
    TOOLS=(userinfoforsoundcloud playlistfordeezer)
    EXTRA=()
    EVAL_TOOLS=(userinfoforsoundcloud playlistfordeezer)
    ;;
  d1_surface)
    DATASET="$ROOT/data/processed/p2a3_d1_surface"
    OUTPUT="$ROOT/experiments/stage4_p2a3_d1_surface_s${SEED}"
    mapfile -t TOOLS < <(find "$DATASET" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
    mapfile -t EVAL_TOOLS < <("$PYTHON" -c 'import json,sys; print(*json.load(open(sys.argv[1])), sep="\n")' "$DATASET/eval_tool_ids.json")
    EXTRA=(--latent-group-manifest "$DATASET/latent_groups.json")
    ;;
  d2_functional)
    DATASET="$ROOT/data/processed/p2a3_d2_functional"
    OUTPUT="$ROOT/experiments/stage4_p2a3_d2_functional_s${SEED}"
    mapfile -t TOOLS < <(find "$DATASET" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
    mapfile -t EVAL_TOOLS < <("$PYTHON" -c 'import json,sys; print(*json.load(open(sys.argv[1])), sep="\n")' "$DATASET/eval_tool_ids.json")
    EXTRA=(--latent-group-manifest "$DATASET/latent_groups.json")
    ;;
  d2_functional_plus_real)
    DATASET="$ROOT/data/processed/p2a3_d2_functional_plus_real"
    OUTPUT="$ROOT/experiments/stage4_p2a3_d2_functional_plus_real_s${SEED}"
    mapfile -t TOOLS < <(find "$DATASET" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
    mapfile -t EVAL_TOOLS < <("$PYTHON" -c 'import json,sys; print(*json.load(open(sys.argv[1])), sep="\n")' "$DATASET/eval_tool_ids.json")
    EXTRA=(--latent-group-manifest "$DATASET/latent_groups.json")
    ;;
  d2_functional_plus_real_native)
    DATASET="$ROOT/data/processed/p2a3_d2_functional_plus_real_native"
    OUTPUT="$ROOT/experiments/stage4_p2a3_d2_functional_plus_real_native_s${SEED}"
    mapfile -t TOOLS < <(find "$DATASET" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
    mapfile -t EVAL_TOOLS < <("$PYTHON" -c 'import json,sys; print(*json.load(open(sys.argv[1])), sep="\n")' "$DATASET/eval_tool_ids.json")
    EXTRA=(--latent-group-manifest "$DATASET/latent_groups.json")
    ;;
  d3_functional_surface)
    DATASET="$ROOT/data/processed/p2a3_d3_functional_surface"
    OUTPUT="$ROOT/experiments/stage4_p2a3_d3_functional_surface_s${SEED}"
    mapfile -t TOOLS < <(find "$DATASET" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
    mapfile -t EVAL_TOOLS < <("$PYTHON" -c 'import json,sys; print(*json.load(open(sys.argv[1])), sep="\n")' "$DATASET/eval_tool_ids.json")
    EXTRA=(--latent-group-manifest "$DATASET/latent_groups.json")
    ;;
  d3_functional_surface_plus_real)
    DATASET="$ROOT/data/processed/p2a3_d3_functional_surface_plus_real"
    OUTPUT="$ROOT/experiments/stage4_p2a3_d3_functional_surface_plus_real_s${SEED}"
    mapfile -t TOOLS < <(find "$DATASET" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
    mapfile -t EVAL_TOOLS < <("$PYTHON" -c 'import json,sys; print(*json.load(open(sys.argv[1])), sep="\n")' "$DATASET/eval_tool_ids.json")
    EXTRA=(--latent-group-manifest "$DATASET/latent_groups.json")
    ;;
  *)
    echo "unknown condition: $CONDITION" >&2
    exit 2
    ;;
esac

if [[ -f "$OUTPUT/training_summary.json" ]]; then
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/functional_hypernet/train_free_latent_shared.py \
  --model "$ROOT/cache/models/Qwen3-0.6B" \
  --dataset-root "$DATASET" \
  --tool-ids "${TOOLS[@]}" \
  --eval-tool-ids "${EVAL_TOOLS[@]}" \
  --output-dir "$OUTPUT" \
  --device cuda \
  --max-steps 400 \
  --examples-per-tool 2 \
  --tools-per-step 2 \
  --eval-interval 50 \
  --seed "$SEED" \
  --objective delta_norm \
  --lambda-regularization 1000 \
  "${EXTRA[@]}"

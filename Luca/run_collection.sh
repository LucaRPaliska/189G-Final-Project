#!/usr/bin/env bash
# run_collection.sh
#
# Run collect_hidden_states.py for each VLM, then balance all datasets.
# Designed to be run on the AWS GPU instance from inside Luca/.
#
# Usage:
#   cd /path/to/189G-Final-Project/Luca
#   bash run_collection.sh
#
# To run a single model only:
#   bash run_collection.sh qwen
#   bash run_collection.sh phi
#   bash run_collection.sh internvl
#   bash run_collection.sh deepseek
#   bash run_collection.sh minicpm

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Models ───────────────────────────────────────────────────────────────────
QWEN="Qwen/Qwen2.5-VL-3B-Instruct"
PHI="microsoft/Phi-3.5-vision-instruct"
INTERNVL="OpenGVLab/InternVL2-2B"
DEEPSEEK="deepseek-ai/deepseek-vl-1.3b-base"
MINICPM="openbmb/MiniCPM-V-2"

# ── Which models to run (filter by first arg if provided) ────────────────────
RUN_ALL=true
TARGET="${1:-all}"
[[ "$TARGET" != "all" ]] && RUN_ALL=false

run_model() {
    local label="$1"
    local model_id="$2"
    if $RUN_ALL || [[ "$TARGET" == "$label" ]]; then
        echo ""
        echo "======================================================"
        echo " Collecting: $model_id"
        echo "======================================================"
        python collect_hidden_states.py \
            --model "$model_id" \
            --split all \
            --pooling last \
            --few_shot_n 3
    fi
}

# ── Collection ───────────────────────────────────────────────────────────────
run_model "qwen"     "$QWEN"
run_model "phi"      "$PHI"
run_model "internvl" "$INTERNVL"
run_model "deepseek" "$DEEPSEEK"
run_model "minicpm"  "$MINICPM"

# ── Balancing ────────────────────────────────────────────────────────────────
if $RUN_ALL || [[ "$TARGET" == "balance" ]]; then
    echo ""
    echo "======================================================"
    echo " Balancing all collected datasets → Luca/balanced/"
    echo "======================================================"
    # Glob all npz files in Luca/ (not inside balanced/)
    NPZ_FILES=( hidden_states_*.npz )
    if [[ ${#NPZ_FILES[@]} -eq 0 ]]; then
        echo "No hidden_states_*.npz files found. Run collection first."
        exit 1
    fi
    python balance_dataset.py "${NPZ_FILES[@]}"
fi

echo ""
echo "Done. Balanced datasets in: $SCRIPT_DIR/balanced/"

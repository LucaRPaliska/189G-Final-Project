#!/usr/bin/env bash
# run_collection.sh
#
# Runs extract_probe_features.py for each VLM (1000 samples each),
# using EM labeling + output hidden state + few-shot prompting.
# Designed to be run on the AWS/Colab GPU instance from the project root.
#
# Usage:
#   bash run_collection.sh              # run all models
#   bash run_collection.sh qwen         # single model
#   bash run_collection.sh phi
#   bash run_collection.sh internvl
#   bash run_collection.sh deepseek
#   bash run_collection.sh minicpm

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/Gege"

# ── Models ────────────────────────────────────────────────────────────────────
QWEN="Qwen/Qwen2.5-VL-3B-Instruct"
PHI="microsoft/Phi-3.5-vision-instruct"
INTERNVL="OpenGVLab/InternVL2-2B"
DEEPSEEK="deepseek-ai/deepseek-vl-1.3b-base"
MINICPM="openbmb/MiniCPM-V-2"

# ── Shared flags ──────────────────────────────────────────────────────────────
SAMPLES=1000
COMMON="--split all --pooling last --max_samples $SAMPLES --shared_test --balance"

# ── Which models to run ───────────────────────────────────────────────────────
TARGET="${1:-all}"

run_model() {
    local label="$1"
    local model_id="$2"

    if [[ "$TARGET" != "all" && "$TARGET" != "$label" ]]; then
        return
    fi

    echo ""
    echo "======================================================"
    echo " Model: $model_id"
    echo "======================================================"

    python3 extract_probe_features.py \
        --model        "$model_id" \
        --labeling     em \
        --hidden_state output \
        --few_shot_n   3 \
        $COMMON
}

# ── Run ───────────────────────────────────────────────────────────────────────
run_model "qwen"     "$QWEN"
run_model "phi"      "$PHI"
run_model "internvl" "$INTERNVL"
run_model "deepseek" "$DEEPSEEK"
run_model "minicpm"  "$MINICPM"

echo ""
echo "======================================================"
echo " Done. Output files in: $ROOT/Gege/"
echo "======================================================"

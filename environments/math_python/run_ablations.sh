#!/bin/bash
# Run ablation experiments for math-python environment
# Usage: ./run_ablations.sh
#
# Ablates across three inference modes:
# - standard: Multi-turn tool-use with sandboxed Python
# - rlm: RLM with REPL access (no tips)
# - rlm_tips: RLM with environment-specific tips

set -e

# Model configurations
# - OpenAI models: just use the model name (e.g., "gpt-4.1-mini")
# - OpenRouter models: prefix with "openrouter:" (e.g., "openrouter:anthropic/claude-3.5-sonnet")

MODELS=(
    "openrouter:xiaomi/mimo-v2-flash:free"
    "openrouter:moonshotai/kimi-k2-thinking"
    "gpt-5-mini"
)

NUM_EXAMPLES=50
ROLLOUTS=1

# Mode configurations: "standard", "rlm", "rlm_tips"
MODES=("standard" "rlm" "rlm_tips")

uv run vf-install math-python

echo "=== Math Python Ablations ==="
echo "Models: ${MODELS[*]}"
echo "Examples per config: $NUM_EXAMPLES"
echo "Rollouts per example: $ROLLOUTS"
echo "Modes: ${MODES[*]}"
echo ""

for MODEL_SPEC in "${MODELS[@]}"; do
    # Parse provider prefix and set API flags
    if [[ "$MODEL_SPEC" == openrouter:* ]]; then
        MODEL="${MODEL_SPEC#openrouter:}"  # Strip "openrouter:" prefix
        API_FLAGS="-k OPENROUTER_API_KEY -b https://openrouter.ai/api/v1"
    else
        MODEL="$MODEL_SPEC"
        API_FLAGS=""
    fi

    echo "########################################"
    echo "### Model: $MODEL"
    echo "########################################"
    echo ""

    for mode in "${MODES[@]}"; do
        # Convert mode to use_rlm and include_env_tips
        case $mode in
            "standard")
                USE_RLM="false"
                INCLUDE_ENV_TIPS="false"
                ;;
            "rlm")
                USE_RLM="true"
                INCLUDE_ENV_TIPS="false"
                ;;
            "rlm_tips")
                USE_RLM="true"
                INCLUDE_ENV_TIPS="true"
                ;;
        esac

        echo "========================================"
        echo "=== Mode: $mode (use_rlm=$USE_RLM, include_env_tips=$INCLUDE_ENV_TIPS) ==="
        echo "========================================"
        echo ""

        echo "Running: model=$MODEL, mode=$mode"
        uv run vf-eval math-python -n $NUM_EXAMPLES -r $ROLLOUTS -m $MODEL $API_FLAGS -s \
            -a "{\"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"dataset_shuffle\": true, \"dataset_seed\": 42}"

        echo ""
    done
done

echo "=== All ablations complete ==="
echo "Results saved to: environments/math_python/outputs/evals/"
echo "Run 'python environments/math_python/aggregate_results.py' to analyze."

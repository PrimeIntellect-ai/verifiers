#!/bin/bash
# Run ablation experiments for math-python environment
# Usage: ./run_ablations.sh
#
# Ablates across three inference modes:
# - standard: Multi-turn tool-use with sandboxed Python
# - rlm: RLM with REPL access (no tips)
# - rlm_tips: RLM with environment-specific tips

set -e

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODELS=(
    # "deepseek:deepseek/deepseek-v3.2"
    "prime:prime-intellect/intellect-3"
    "openrouter:xiaomi/mimo-v2-flash:free"
    "openrouter:z-ai/glm-4.5-air"
    "openrouter:z-ai/glm-4.6"
)

NUM_EXAMPLES=50
ROLLOUTS=1

# Mode configurations: "standard", "rlm", "rlm_tips"
MODES=("rlm" "rlm_tips" "standard")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

run_eval_deepseek() {
    local MODEL="$1"
    local USE_RLM="$2"
    local INCLUDE_ENV_TIPS="$3"
    
    uv run vf-eval math-python -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" \
        -k OPENROUTER_API_KEY -b https://openrouter.ai/api/v1 \
        -S '{"extra_body": {"reasoning": {"enabled": true}}}' \
        -s -a "{\"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"shuffle\": true, \"seed\": 42}"
}

run_eval_prime() {
    local MODEL="$1"
    local USE_RLM="$2"
    local INCLUDE_ENV_TIPS="$3"
    
    uv run vf-eval math-python -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" \
        -k PRIME_API_KEY -b https://api.pinference.ai/api/v1 \
        --header 'X-Prime-Team-ID: clyvldofb0000gg1kx39rgzjq' \
        -s -a "{\"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"shuffle\": true, \"seed\": 42}"
}

run_eval_openrouter() {
    local MODEL="$1"
    local USE_RLM="$2"
    local INCLUDE_ENV_TIPS="$3"
    
    uv run vf-eval math-python -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" \
        -k OPENROUTER_API_KEY -b https://openrouter.ai/api/v1 \
        -s -a "{\"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"shuffle\": true, \"seed\": 42}"
}

run_eval_openai() {
    local MODEL="$1"
    local USE_RLM="$2"
    local INCLUDE_ENV_TIPS="$3"
    
    uv run vf-eval math-python -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" \
        -s -a "{\"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"shuffle\": true, \"seed\": 42}"
}

run_model() {
    local MODEL_SPEC="$1"
    local USE_RLM="$2"
    local INCLUDE_ENV_TIPS="$3"
    
    if [[ "$MODEL_SPEC" == deepseek:* ]]; then
        local MODEL="${MODEL_SPEC#deepseek:}"
        run_eval_deepseek "$MODEL" "$USE_RLM" "$INCLUDE_ENV_TIPS"
    elif [[ "$MODEL_SPEC" == prime:* ]]; then
        local MODEL="${MODEL_SPEC#prime:}"
        run_eval_prime "$MODEL" "$USE_RLM" "$INCLUDE_ENV_TIPS"
    elif [[ "$MODEL_SPEC" == openrouter:* ]]; then
        local MODEL="${MODEL_SPEC#openrouter:}"
        run_eval_openrouter "$MODEL" "$USE_RLM" "$INCLUDE_ENV_TIPS"
    else
        run_eval_openai "$MODEL_SPEC" "$USE_RLM" "$INCLUDE_ENV_TIPS"
    fi
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

uv run vf-install math-python

echo "=== Math Python Ablations ==="
echo "Models: ${MODELS[*]}"
echo "Examples per config: $NUM_EXAMPLES"
echo "Rollouts per example: $ROLLOUTS"
echo "Modes: ${MODES[*]}"
echo ""

for MODEL_SPEC in "${MODELS[@]}"; do
    echo "########################################"
    echo "### Model: $MODEL_SPEC"
    echo "########################################"
    echo ""

    for mode in "${MODES[@]}"; do
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

        echo "Running: model=$MODEL_SPEC, mode=$mode"
        run_model "$MODEL_SPEC" "$USE_RLM" "$INCLUDE_ENV_TIPS"
        echo ""
    done
done

echo "=== All ablations complete ==="
echo "Results saved to: environments/math_python/outputs/evals/"
echo "Run 'python environments/math_python/aggregate_results.py' to analyze."

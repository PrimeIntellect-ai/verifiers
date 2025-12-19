#!/bin/bash
# Run ablation experiments for deepdive environment
# Usage: ./run_ablations.sh
#
# Ablates across three inference modes:
# - standard: Multi-turn tool-use LLM (direct search/open tools)
# - rlm: RLM with REPL access, sub-LLMs have tools (no tips)
# - rlm_tips: RLM with environment-specific tips
#
# Model groups:
# - MODELS_FULL: Run all mode ablations (deepseek, intellect-3)
# - MODELS_STANDARD: Run only default mode=rlm_tips (broader model coverage)

set -e

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# MODELS_FULL: These models run ALL ablations (all modes)
# Used for comprehensive testing with our core models
MODELS_FULL=(
    "deepseek:deepseek/deepseek-v3.2"
    "prime:prime-intellect/intellect-3"
)

# MODELS_STANDARD: These models run only the default setting (mode=rlm_tips)
# Used for broader model coverage without full ablation cost
MODELS_STANDARD=(
    "openrouter:xiaomi/mimo-v2-flash:free"
    "openrouter:z-ai/glm-4.5-air"
    "openrouter:z-ai/glm-4.6"
)

NUM_EXAMPLES=100
ROLLOUTS=1

# Mode configurations: "standard", "rlm", "rlm_tips"
MODES=("rlm" "rlm_tips" "standard")

# Default mode for MODELS_STANDARD
DEFAULT_MODE="rlm_tips"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

run_eval_deepseek() {
    local MODEL="$1"
    local USE_RLM="$2"
    local INCLUDE_ENV_TIPS="$3"
    
    uv run vf-eval deepdive -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" \
        -k OPENROUTER_API_KEY -b https://openrouter.ai/api/v1 \
        -S '{"extra_body": {"reasoning": {"enabled": true}}}' \
        -s -a "{\"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"shuffle\": true, \"seed\": 42}"
}

run_eval_prime() {
    local MODEL="$1"
    local USE_RLM="$2"
    local INCLUDE_ENV_TIPS="$3"
    
    uv run vf-eval deepdive -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" \
        -k PRIME_API_KEY -b https://api.pinference.ai/api/v1 \
        --header 'X-Prime-Team-ID: clyvldofb0000gg1kx39rgzjq' \
        -s -a "{\"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"shuffle\": true, \"seed\": 42}"
}

run_eval_openrouter() {
    local MODEL="$1"
    local USE_RLM="$2"
    local INCLUDE_ENV_TIPS="$3"
    
    uv run vf-eval deepdive -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" \
        -k OPENROUTER_API_KEY -b https://openrouter.ai/api/v1 \
        -s -a "{\"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"shuffle\": true, \"seed\": 42}"
}

run_eval_openai() {
    local MODEL="$1"
    local USE_RLM="$2"
    local INCLUDE_ENV_TIPS="$3"
    
    uv run vf-eval deepdive -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" \
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

uv run vf-install deepdive

echo "=== DeepDive Ablations ==="
echo "MODELS_FULL (all modes): ${MODELS_FULL[*]}"
echo "MODELS_STANDARD (default mode only): ${MODELS_STANDARD[*]}"
echo "Examples per config: $NUM_EXAMPLES"
echo "Rollouts per example: $ROLLOUTS"
echo "Modes for MODELS_FULL: ${MODES[*]}"
echo "Default mode for MODELS_STANDARD: $DEFAULT_MODE"
echo ""

# -----------------------------------------------------------------------------
# PART 1: Full ablations with MODELS_FULL
# -----------------------------------------------------------------------------
echo "############################################################"
echo "### PART 1: Full ablations with MODELS_FULL"
echo "############################################################"
echo ""

for MODEL_SPEC in "${MODELS_FULL[@]}"; do
    echo "########################################"
    echo "### Model: $MODEL_SPEC (full ablation)"
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

# -----------------------------------------------------------------------------
# PART 2: Default setting only with MODELS_STANDARD
# -----------------------------------------------------------------------------
echo "############################################################"
echo "### PART 2: Default setting (mode=$DEFAULT_MODE) with MODELS_STANDARD"
echo "############################################################"
echo ""

# Set default mode flags
USE_RLM="true"
INCLUDE_ENV_TIPS="true"

for MODEL_SPEC in "${MODELS_STANDARD[@]}"; do
    echo "########################################"
    echo "### Model: $MODEL_SPEC (default setting only)"
    echo "########################################"
    echo ""

    echo "Running: model=$MODEL_SPEC, mode=$DEFAULT_MODE"
    run_model "$MODEL_SPEC" "$USE_RLM" "$INCLUDE_ENV_TIPS"
    echo ""
done

echo "=== All ablations complete ==="
echo "Results saved to: environments/deepdive/outputs/evals/"
echo "Run 'uv run environments/deepdive/aggregate_results.py' to analyze."

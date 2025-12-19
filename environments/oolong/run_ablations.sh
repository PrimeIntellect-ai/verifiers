#!/bin/bash
# Run ablation experiments for oolong long-context environment
# Usage: ./run_ablations.sh
#
# Ablates across:
# - Three inference modes: standard, rlm, rlm_tips
# - Three subsets: synth, synth_with_labels, real
#
# Model groups:
# - MODELS_FULL: Run all ablations (deepseek, intellect-3)
# - MODELS_STANDARD: Run only default setting (broader model coverage)
#   Default: mode=rlm_tips, subset=real

set -e

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# MODELS_FULL: These models run ALL ablations
# Used for comprehensive testing with our core models
MODELS_FULL=(
    # "deepseek:deepseek/deepseek-v3.2"
    "prime:prime-intellect/intellect-3"
)

# MODELS_STANDARD: These models run only the default setting
# Used for broader model coverage without full ablation cost
MODELS_STANDARD=(
    "openrouter:xiaomi/mimo-v2-flash:free"
    "openrouter:z-ai/glm-4.5-air"
    "openrouter:z-ai/glm-4.6"
)

# Fewer examples since long-context evaluation is slow/costly
NUM_EXAMPLES=50
ROLLOUTS=1

# Mode configurations: "standard", "rlm", "rlm_tips"
MODES=("rlm" "rlm_tips" "standard")

# Subset configurations: "synth", "synth_with_labels", "real"
SUBSETS=("synth" "synth_with_labels" "real")

# Default settings for MODELS_STANDARD
DEFAULT_MODE="rlm_tips"
DEFAULT_SUBSET="real"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

run_eval_deepseek() {
    local MODEL="$1"
    local ENV_ARGS="$2"
    
    uv run vf-eval oolong -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" \
        -k OPENROUTER_API_KEY -b https://openrouter.ai/api/v1 \
        -S '{"extra_body": {"reasoning": {"enabled": true}}}' \
        -s -a "$ENV_ARGS"
}

run_eval_prime() {
    local MODEL="$1"
    local ENV_ARGS="$2"
    
    uv run vf-eval oolong -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" \
        -k PRIME_API_KEY -b https://api.pinference.ai/api/v1 \
        --header 'X-Prime-Team-ID: clyvldofb0000gg1kx39rgzjq' \
        -s -a "$ENV_ARGS"
}

run_eval_openrouter() {
    local MODEL="$1"
    local ENV_ARGS="$2"
    
    uv run vf-eval oolong -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" \
        -k OPENROUTER_API_KEY -b https://openrouter.ai/api/v1 \
        -s -a "$ENV_ARGS"
}

run_eval_openai() {
    local MODEL="$1"
    local ENV_ARGS="$2"
    
    uv run vf-eval oolong -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" \
        -s -a "$ENV_ARGS"
}

run_model() {
    local MODEL_SPEC="$1"
    local ENV_ARGS="$2"
    
    if [[ "$MODEL_SPEC" == deepseek:* ]]; then
        local MODEL="${MODEL_SPEC#deepseek:}"
        run_eval_deepseek "$MODEL" "$ENV_ARGS"
    elif [[ "$MODEL_SPEC" == prime:* ]]; then
        local MODEL="${MODEL_SPEC#prime:}"
        run_eval_prime "$MODEL" "$ENV_ARGS"
    elif [[ "$MODEL_SPEC" == openrouter:* ]]; then
        local MODEL="${MODEL_SPEC#openrouter:}"
        run_eval_openrouter "$MODEL" "$ENV_ARGS"
    else
        run_eval_openai "$MODEL_SPEC" "$ENV_ARGS"
    fi
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

uv run vf-install oolong

echo "=== Oolong Long-Context Ablations ==="
echo "MODELS_FULL (all ablations): ${MODELS_FULL[*]}"
echo "MODELS_STANDARD (default setting only): ${MODELS_STANDARD[*]}"
echo "Examples per config: $NUM_EXAMPLES"
echo "Rollouts per example: $ROLLOUTS"
echo "Modes for MODELS_FULL: ${MODES[*]}"
echo "Subsets for MODELS_FULL: ${SUBSETS[*]}"
echo "Default for MODELS_STANDARD: mode=$DEFAULT_MODE, subset=$DEFAULT_SUBSET"
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

        for subset in "${SUBSETS[@]}"; do
            echo "Running: model=$MODEL_SPEC, mode=$mode, subset=$subset"
            run_model "$MODEL_SPEC" "{\"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"subset\": \"$subset\", \"shuffle\": true, \"seed\": 42}"
            echo ""
        done
    done
done

# -----------------------------------------------------------------------------
# PART 2: Default setting only with MODELS_STANDARD
# -----------------------------------------------------------------------------
echo "############################################################"
echo "### PART 2: Default setting with MODELS_STANDARD"
echo "############################################################"
echo ""

# Set default mode flags (rlm_tips)
USE_RLM="true"
INCLUDE_ENV_TIPS="true"

for MODEL_SPEC in "${MODELS_STANDARD[@]}"; do
    echo "########################################"
    echo "### Model: $MODEL_SPEC (default setting only)"
    echo "########################################"
    echo ""

    echo "Running: model=$MODEL_SPEC, mode=$DEFAULT_MODE, subset=$DEFAULT_SUBSET"
    run_model "$MODEL_SPEC" "{\"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"subset\": \"$DEFAULT_SUBSET\", \"shuffle\": true, \"seed\": 42}"
    echo ""
done

echo "=== All ablations complete ==="
echo "Results saved to: environments/oolong/outputs/evals/"
echo "Run 'python environments/oolong/aggregate_results.py' to analyze."

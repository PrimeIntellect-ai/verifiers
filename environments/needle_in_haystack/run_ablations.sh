#!/bin/bash
# Run ablation experiments for needle-in-haystack environment
# Usage: ./run_ablations.sh
#
# Ablates across:
# - Three inference modes: standard, rlm, rlm_tips
# - Two needle types: word, numeric
# - Four context sizes: 1000, 5000, 10000, 20000 lines
# - Three needle counts: 1, 3, 5
#
# Model groups:
# - MODELS_FULL: Run all ablations (all modes × all ablation dimensions)
# - MODELS_STANDARD: Run all modes but only default ablation settings (broader model coverage)

set -e

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# MODELS_FULL: These models run ALL ablations
# Used for comprehensive testing with our core models
MODELS_FULL=(
    # "prime:prime-intellect/intellect-3"
    "gpt-5-mini"
)

# MODELS_STANDARD: These models run all modes but only default ablation settings
# Used for broader model coverage without full ablation cost
MODELS_STANDARD=(
    # "openrouter:xiaomi/mimo-v2-flash:free"
    # "openrouter:z-ai/glm-4.6"
    # "deepseek:deepseek/deepseek-v3.2"
    # "openrouter:z-ai/glm-4.5-air"
)

NUM_EXAMPLES=50
ROLLOUTS=1
CONCURRENCY=50

# Mode configurations: "standard", "rlm", "rlm_tips"
MODES=("rlm" "rlm_tips" "standard")

# Default ablation settings for MODELS_STANDARD
DEFAULT_NEEDLE_TYPE="word"
DEFAULT_NUM_LINES=10000
DEFAULT_NUM_NEEDLES=1

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

run_eval_deepseek() {
    local MODEL="$1"
    local ENV_ARGS="$2"
    
    uv run vf-eval needle-in-haystack -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" -c $CONCURRENCY \
        -k OPENROUTER_API_KEY -b https://openrouter.ai/api/v1 \
        -S '{"extra_body": {"reasoning": {"enabled": true}, "provider": {"only": ["google-vertex"], "allow_fallbacks": false, "require_parameters": true}}}' \
        -s -a "$ENV_ARGS"
}

run_eval_prime() {
    local MODEL="$1"
    local ENV_ARGS="$2"
    
    uv run vf-eval needle-in-haystack -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" -c $CONCURRENCY \
        -k PRIME_API_KEY -b https://api.pinference.ai/api/v1 \
        --header 'X-Prime-Team-ID: clyvldofb0000gg1kx39rgzjq' \
        -s -a "$ENV_ARGS"
}

run_eval_openrouter() {
    local MODEL="$1"
    local ENV_ARGS="$2"
    
    # Determine provider based on model
    local PROVIDER_JSON=""
    if [[ "$MODEL" == xiaomi/* ]]; then
        PROVIDER_JSON='"provider": {"only": ["xiaomi/fp8"], "allow_fallbacks": false, "require_parameters": true}'
    elif [[ "$MODEL" == z-ai/* ]]; then
        PROVIDER_JSON='"provider": {"only": ["z-ai"], "allow_fallbacks": false, "require_parameters": true}'
    fi
    
    uv run vf-eval needle-in-haystack -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" -c $CONCURRENCY \
        -k OPENROUTER_API_KEY -b https://openrouter.ai/api/v1 \
        -S "{\"extra_body\": {$PROVIDER_JSON}}" \
        -s -a "$ENV_ARGS"
}

run_eval_openai() {
    local MODEL="$1"
    local ENV_ARGS="$2"
    
    uv run vf-eval needle-in-haystack -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" -c $CONCURRENCY \
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

uv run vf-install needle-in-haystack

echo "=== Needle in Haystack Ablations ==="
echo "MODELS_FULL (all ablations): ${MODELS_FULL[*]}"
echo "MODELS_STANDARD (all modes, default ablation settings): ${MODELS_STANDARD[*]}"
echo "Examples per config: $NUM_EXAMPLES"
echo "Rollouts per example: $ROLLOUTS"
echo "Modes for MODELS_FULL: ${MODES[*]}"
echo "Default ablation settings for MODELS_STANDARD: needle_type=$DEFAULT_NEEDLE_TYPE, num_lines=$DEFAULT_NUM_LINES, num_needles=$DEFAULT_NUM_NEEDLES"
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

        # Ablation 1: Needle type effect
        # Fixed: num_lines=10000, num_needles=1
        echo "=== Ablation 1: Needle Type ($mode) ==="
        for needle_type in word numeric; do
            echo "Running: mode=$mode, needle_type=$needle_type"
            run_model "$MODEL_SPEC" "{\"num_samples\": $NUM_EXAMPLES, \"needle_type\": \"$needle_type\", \"num_lines\": 10000, \"num_needles\": 1, \"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"shuffle\": true, \"seed\": 42}"
        done

        # Ablation 2: Context size scaling
        # Fixed: needle_type="word", num_needles=1
        echo ""
        echo "=== Ablation 2: Context Size ($mode) ==="
        for num_lines in 1000 5000 10000 20000; do
            echo "Running: mode=$mode, num_lines=$num_lines"
            run_model "$MODEL_SPEC" "{\"num_samples\": $NUM_EXAMPLES, \"needle_type\": \"word\", \"num_lines\": $num_lines, \"num_needles\": 1, \"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"shuffle\": true, \"seed\": 42}"
        done

        # Ablation 3: Multi-needle scaling
        # Fixed: needle_type="word", num_lines=10000
        echo ""
        echo "=== Ablation 3: Needle Count ($mode) ==="
        for num_needles in 1 3 5; do
            echo "Running: mode=$mode, num_needles=$num_needles"
            run_model "$MODEL_SPEC" "{\"num_samples\": $NUM_EXAMPLES, \"needle_type\": \"word\", \"num_lines\": 10000, \"num_needles\": $num_needles, \"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"shuffle\": true, \"seed\": 42}"
        done

        echo ""
    done
done

# -----------------------------------------------------------------------------
# PART 2: All modes with MODELS_STANDARD (but only default ablation settings)
# -----------------------------------------------------------------------------
echo "############################################################"
echo "### PART 2: All modes with MODELS_STANDARD (default ablation settings)"
echo "############################################################"
echo ""

for MODEL_SPEC in "${MODELS_STANDARD[@]}"; do
    echo "########################################"
    echo "### Model: $MODEL_SPEC (all modes, default ablation settings)"
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

        echo "Running: model=$MODEL_SPEC, mode=$mode, needle_type=$DEFAULT_NEEDLE_TYPE, num_lines=$DEFAULT_NUM_LINES, num_needles=$DEFAULT_NUM_NEEDLES"
        run_model "$MODEL_SPEC" "{\"num_samples\": $NUM_EXAMPLES, \"needle_type\": \"$DEFAULT_NEEDLE_TYPE\", \"num_lines\": $DEFAULT_NUM_LINES, \"num_needles\": $DEFAULT_NUM_NEEDLES, \"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"shuffle\": true, \"seed\": 42}"
        echo ""
    done
done

echo "=== All ablations complete ==="
echo "Results saved to: environments/needle_in_haystack/outputs/evals/"
echo "Run 'python environments/needle_in_haystack/aggregate_results.py' to analyze."

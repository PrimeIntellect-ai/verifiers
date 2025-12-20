#!/bin/bash
# Run ablation experiments for verbatim_copy environment
# Usage: ./run_ablations.sh
#
# Ablates across three inference modes:
# - standard: Single-turn LLM generation
# - rlm: RLM with REPL access (no tips)
# - rlm_tips: RLM with environment-specific tips
#
# And three ablation dimensions:
# - content_type: words, json, csv, codes, mixed, all
# - target_length: 250, 500, 750, 1000
# - mean_fragment_length: null, 10, 25, 50, 100, 150
#
# Model groups:
# - MODELS_FULL: Run all ablations (deepseek, intellect-3)
# - MODELS_STANDARD: Run only default setting (broader model coverage)
#   Default: mode=rlm_tips, content_type=all, target_length=500, mean_fragment_length=20

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

# MODELS_STANDARD: These models run only the default setting
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

# Default settings for MODELS_STANDARD
DEFAULT_MODE="rlm_tips"
DEFAULT_CONTENT_TYPE="all"
DEFAULT_TARGET_LENGTH=500
DEFAULT_FRAGMENT_LENGTH=20

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

run_eval_deepseek() {
    local MODEL="$1"
    local ENV_ARGS="$2"
    
    uv run vf-eval verbatim-copy -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" -c $CONCURRENCY \
        -k OPENROUTER_API_KEY -b https://openrouter.ai/api/v1 \
        -S '{"extra_body": {"reasoning": {"enabled": true}, "provider": {"only": ["google-vertex"], "allow_fallbacks": false, "require_parameters": true}}}' \
        -s -a "$ENV_ARGS"
}

run_eval_prime() {
    local MODEL="$1"
    local ENV_ARGS="$2"
    
    uv run vf-eval verbatim-copy -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" -c $CONCURRENCY \
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
    
    uv run vf-eval verbatim-copy -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" -c $CONCURRENCY \
        -k OPENROUTER_API_KEY -b https://openrouter.ai/api/v1 \
        -S "{\"extra_body\": {$PROVIDER_JSON}}" \
        -s -a "$ENV_ARGS"
}

run_eval_openai() {
    local MODEL="$1"
    local ENV_ARGS="$2"
    
    uv run vf-eval verbatim-copy -n $NUM_EXAMPLES -r $ROLLOUTS -m "$MODEL" -c $CONCURRENCY \
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

uv run vf-install verbatim-copy

echo "=== Verbatim Copy Ablations ==="
echo "MODELS_FULL (all ablations): ${MODELS_FULL[*]}"
echo "MODELS_STANDARD (default setting only): ${MODELS_STANDARD[*]}"
echo "Examples per config: $NUM_EXAMPLES"
echo "Rollouts per example: $ROLLOUTS"
echo "Modes for MODELS_FULL: ${MODES[*]}"
echo "Default for MODELS_STANDARD: mode=$DEFAULT_MODE, content_type=$DEFAULT_CONTENT_TYPE, target_length=$DEFAULT_TARGET_LENGTH, fragment_length=$DEFAULT_FRAGMENT_LENGTH"
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

        # Ablation 1: Content type effect
        # Fixed: target_length=500, mean_fragment_length=20
        echo "=== Ablation 1: Content Type ($mode) ==="
        for content_type in words json csv codes mixed all; do
            echo "Running: mode=$mode, content_type=$content_type"
            run_model "$MODEL_SPEC" "{\"num_samples\": $NUM_EXAMPLES, \"content_type\": \"$content_type\", \"target_length\": 500, \"mean_fragment_length\": 20, \"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"shuffle\": true, \"seed\": 42}"
        done

        # Ablation 2: Length scaling
        # Fixed: content_type="all", mean_fragment_length=20
        echo ""
        echo "=== Ablation 2: Target Length ($mode) ==="
        for length in 250 500 750 1000; do
            echo "Running: mode=$mode, target_length=$length"
            run_model "$MODEL_SPEC" "{\"num_samples\": $NUM_EXAMPLES, \"content_type\": \"all\", \"target_length\": $length, \"mean_fragment_length\": 20, \"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"shuffle\": true, \"seed\": 42}"
        done

        # Ablation 3: Fragmentation intensity
        # Fixed: content_type="all", target_length=500
        echo ""
        echo "=== Ablation 3: Fragment Length ($mode) ==="
        for frag_length in null 10 25 50 100 150; do
            echo "Running: mode=$mode, mean_fragment_length=$frag_length"
            run_model "$MODEL_SPEC" "{\"num_samples\": $NUM_EXAMPLES, \"content_type\": \"all\", \"target_length\": 500, \"mean_fragment_length\": $frag_length, \"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"shuffle\": true, \"seed\": 42}"
        done

        echo ""
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

    echo "Running: model=$MODEL_SPEC, mode=$DEFAULT_MODE, content_type=$DEFAULT_CONTENT_TYPE, target_length=$DEFAULT_TARGET_LENGTH, fragment_length=$DEFAULT_FRAGMENT_LENGTH"
    run_model "$MODEL_SPEC" "{\"num_samples\": $NUM_EXAMPLES, \"content_type\": \"$DEFAULT_CONTENT_TYPE\", \"target_length\": $DEFAULT_TARGET_LENGTH, \"mean_fragment_length\": $DEFAULT_FRAGMENT_LENGTH, \"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"shuffle\": true, \"seed\": 42}"
    echo ""
done

echo "=== All ablations complete ==="
echo "Results saved to: environments/verbatim_copy/outputs/evals/"
echo "Run 'python environments/verbatim_copy/aggregate_results.py' to analyze."

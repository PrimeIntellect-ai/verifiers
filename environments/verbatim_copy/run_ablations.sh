#!/bin/bash
# Run ablation experiments for verbatim_copy environment
# Usage: ./run_ablations.sh
#
# Ablates across three inference modes:
# - standard: Single-turn LLM generation
# - rlm: RLM with REPL access (no tips)
# - rlm_tips: RLM with environment-specific tips

set -e

# Model configurations
# - OpenAI models: just use the model name (e.g., "gpt-5-mini")
# - OpenRouter models: prefix with "openrouter:" (e.g., "openrouter:xiaomi/mimo-v2-flash:free")
# Examples:
#   "gpt-5-mini"
#   "openrouter:anthropic/claude-3-5-sonnet"
#   "openrouter:xiaomi/mimo-v2-flash:free"
MODELS=(
    "openrouter:xiaomi/mimo-v2-flash:free"
    "openrouter:allenai/olmo-3.1-32b-think:free"
    "openrouter:moonshotai/kimi-k2-thinking"
    "gpt-5-mini"
)

NUM_EXAMPLES=200
ROLLOUTS=1

# Mode configurations: "standard", "rlm", "rlm_tips"
# Comment out modes to skip them
MODES=("rlm" "rlm_tips" "standard")

uv run vf-install verbatim-copy

echo "=== Verbatim Copy Ablations ==="
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

    # Ablation 1: Content type effect
    # Fixed: target_length=500, mean_fragment_length=20
    echo "=== Ablation 1: Content Type ($mode) ==="
    for content_type in words json csv codes mixed all; do
        echo "Running: mode=$mode, content_type=$content_type"
        uv run vf-eval verbatim-copy -n $NUM_EXAMPLES -r $ROLLOUTS -m $MODEL $API_FLAGS -s \
            -a "{\"num_samples\": $NUM_EXAMPLES, \"content_type\": \"$content_type\", \"target_length\": 500, \"mean_fragment_length\": 20, \"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS}"
    done

    # Ablation 2: Length scaling
    # Fixed: content_type="all", mean_fragment_length=20
    echo ""
    echo "=== Ablation 2: Target Length ($mode) ==="
    for length in 250 500 750 1000; do
        echo "Running: mode=$mode, target_length=$length"
        uv run vf-eval verbatim-copy -n $NUM_EXAMPLES -r $ROLLOUTS -m $MODEL $API_FLAGS -s \
            -a "{\"num_samples\": $NUM_EXAMPLES, \"content_type\": \"all\", \"target_length\": $length, \"mean_fragment_length\": 20, \"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS}"
    done

    # Ablation 3: Fragmentation intensity
    # Fixed: content_type="all", target_length=500
    echo ""
    echo "=== Ablation 3: Fragment Length ($mode) ==="
    # Note: "null" in JSON for None
    for frag_length in null 10 25 50 100 150; do
        echo "Running: mode=$mode, mean_fragment_length=$frag_length"
        uv run vf-eval verbatim-copy -n $NUM_EXAMPLES -r $ROLLOUTS -m $MODEL $API_FLAGS -s \
            -a "{\"num_samples\": $NUM_EXAMPLES, \"content_type\": \"all\", \"target_length\": 500, \"mean_fragment_length\": $frag_length, \"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS}"
    done

    echo ""
done

done  # End of MODELS loop

echo "=== All ablations complete ==="
echo "Results saved to: environments/verbatim_copy/outputs/evals/"
echo "Run 'python environments/verbatim_copy/aggregate_results.py' to analyze."

#!/bin/bash
# Run ablation experiments for needle-in-haystack environment
# Usage: ./run_ablations.sh
#
# Ablates across:
# - Three inference modes: standard, rlm, rlm_tips
# - Two needle types: word, numeric
# - Four context sizes: 1000, 5000, 10000, 20000 lines
# - Three needle counts: 1, 3, 5

set -e

# Model configurations
# - OpenAI models: just use the model name (e.g., "gpt-4.1-mini")
# - OpenRouter models: prefix with "openrouter:" (e.g., "openrouter:anthropic/claude-3.5-sonnet")

MODELS=(
    "openrouter:xiaomi/mimo-v2-flash:free"
    "openrouter:moonshotai/kimi-k2-thinking"
    "gpt-5-mini"
)

NUM_EXAMPLES=200
ROLLOUTS=1

# Mode configurations: "standard", "rlm", "rlm_tips"
MODES=("rlm" "rlm_tips" "standard")

uv run vf-install needle-in-haystack

echo "=== Needle in Haystack Ablations ==="
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

        # Ablation 1: Needle type effect
        # Fixed: num_lines=10000, num_needles=1
        echo "=== Ablation 1: Needle Type ($mode) ==="
        for needle_type in word numeric; do
            echo "Running: mode=$mode, needle_type=$needle_type"
            uv run vf-eval needle-in-haystack -n $NUM_EXAMPLES -r $ROLLOUTS -m $MODEL $API_FLAGS -s \
                -a "{\"num_samples\": $NUM_EXAMPLES, \"needle_type\": \"$needle_type\", \"num_lines\": 10000, \"num_needles\": 1, \"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"seed\": 42}"
        done

        # Ablation 2: Context size scaling
        # Fixed: needle_type="word", num_needles=1
        echo ""
        echo "=== Ablation 2: Context Size ($mode) ==="
        for num_lines in 1000 5000 10000 20000; do
            echo "Running: mode=$mode, num_lines=$num_lines"
            uv run vf-eval needle-in-haystack -n $NUM_EXAMPLES -r $ROLLOUTS -m $MODEL $API_FLAGS -s \
                -a "{\"num_samples\": $NUM_EXAMPLES, \"needle_type\": \"word\", \"num_lines\": $num_lines, \"num_needles\": 1, \"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"seed\": 42}"
        done

        # Ablation 3: Multi-needle scaling
        # Fixed: needle_type="word", num_lines=10000
        echo ""
        echo "=== Ablation 3: Needle Count ($mode) ==="
        for num_needles in 1 3 5; do
            echo "Running: mode=$mode, num_needles=$num_needles"
            uv run vf-eval needle-in-haystack -n $NUM_EXAMPLES -r $ROLLOUTS -m $MODEL $API_FLAGS -s \
                -a "{\"num_samples\": $NUM_EXAMPLES, \"needle_type\": \"word\", \"num_lines\": 10000, \"num_needles\": $num_needles, \"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"seed\": 42}"
        done

        echo ""
    done
done

echo "=== All ablations complete ==="
echo "Results saved to: environments/needle_in_haystack/outputs/evals/"
echo "Run 'python environments/needle_in_haystack/aggregate_results.py' to analyze."

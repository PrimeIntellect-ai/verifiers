#!/bin/bash
# Run ablation experiments for oolong long-context environment
# Usage: ./run_ablations.sh
#
# Ablates across:
# - Three inference modes: standard, rlm, rlm_tips
# - Three subsets: synth, synth_with_labels, real

set -e

# Model configurations
# - OpenAI models: just use the model name (e.g., "gpt-4.1-mini")
# - OpenRouter models: prefix with "openrouter:" (e.g., "openrouter:anthropic/claude-3.5-sonnet")
MODELS=(
    "gpt-4.1-mini"
    "gpt-4.1"
)

# Fewer examples since long-context evaluation is slow/costly
NUM_EXAMPLES=200
ROLLOUTS=1

# Mode configurations: "standard", "rlm", "rlm_tips"
MODES=("rlm" "rlm_tips" "standard")

# Subset configurations: "synth", "synth_with_labels", "real"
SUBSETS=("synth" "synth_with_labels" "real")

uv run vf-install oolong

echo "=== Oolong Long-Context Ablations ==="
echo "Models: ${MODELS[*]}"
echo "Examples per config: $NUM_EXAMPLES"
echo "Rollouts per example: $ROLLOUTS"
echo "Modes: ${MODES[*]}"
echo "Subsets: ${SUBSETS[*]}"
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

        for subset in "${SUBSETS[@]}"; do
            echo "Running: model=$MODEL, mode=$mode, subset=$subset"
            uv run vf-eval oolong -n $NUM_EXAMPLES -r $ROLLOUTS -m $MODEL $API_FLAGS -s \
                -a "{\"use_rlm\": $USE_RLM, \"include_env_tips\": $INCLUDE_ENV_TIPS, \"subset\": \"$subset\"}"
            echo ""
        done
    done
done

echo "=== All ablations complete ==="
echo "Results saved to: environments/oolong/outputs/evals/"
echo "Run 'python environments/oolong/aggregate_results.py' to analyze."

#!/bin/bash
# Run ablation experiments for verbatim_copy environment
# Usage: ./run_ablations.sh

set -e

MODEL="gpt-5-mini"
NUM_EXAMPLES=200
ROLLOUTS=1

uv run vf-install verbatim_copy

echo "=== Verbatim Copy Ablations ==="
echo "Model: $MODEL"
echo "Examples per config: $NUM_EXAMPLES"
echo "Rollouts per example: $ROLLOUTS"
echo ""

# Ablation 1: Data complexity effect
# Fixed: target_length=500, mean_fragment_length=20
echo "=== Ablation 1: Data Complexity ==="
for complexity in words structured codes mixed all; do
    echo "Running: data_complexity=$complexity"
    uv run vf-eval verbatim_copy -n $NUM_EXAMPLES -r $ROLLOUTS -m $MODEL -s \
        -a "{\"num_samples\": $NUM_EXAMPLES, \"data_complexity\": \"$complexity\", \"target_length\": 500, \"mean_fragment_length\": 20, \"use_rlm\": false}"
done

# Ablation 2: Length scaling
# Fixed: data_complexity="all", mean_fragment_length=20
echo ""
echo "=== Ablation 2: Target Length ==="
for length in 250 500 750 1000; do
    echo "Running: target_length=$length"
    uv run vf-eval verbatim_copy -n $NUM_EXAMPLES -r $ROLLOUTS -m $MODEL -s \
        -a "{\"num_samples\": $NUM_EXAMPLES, \"data_complexity\": \"all\", \"target_length\": $length, \"mean_fragment_length\": 20, \"use_rlm\": false}"
done

# Ablation 3: Fragmentation intensity
# Fixed: data_complexity="all", target_length=500
echo ""
echo "=== Ablation 3: Fragment Length ==="
# Note: "null" in JSON for None
for frag_length in null 10 25 50 100 150; do
    echo "Running: mean_fragment_length=$frag_length"
    uv run vf-eval verbatim_copy -n $NUM_EXAMPLES -r $ROLLOUTS -m $MODEL -s \
        -a "{\"num_samples\": $NUM_EXAMPLES, \"data_complexity\": \"all\", \"target_length\": 500, \"mean_fragment_length\": $frag_length, \"use_rlm\": false}"
done

echo ""
echo "=== All ablations complete ==="
echo "Results saved to: environments/verbatim_copy/outputs/evals/"
echo "Run 'python environments/verbatim_copy/aggregate_results.py' to analyze."

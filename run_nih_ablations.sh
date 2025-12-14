#!/bin/bash
set -euo pipefail

MODEL="gpt-5-mini"
NUM_EXAMPLES=10
ROLLOUTS=1
SEED=42
METRICS_FILE="needle_metrics.json"

# Ablation dimensions
POSITIONS=(0.125 0.375 0.625 0.875)
VARIANCE=0.125
NUM_LINES=(1024 2048 4096 8192 16384 32768)
NUM_NEEDLES=(1 2 3)
NEEDLE_TYPE="word"  # Use word-based needles (harder)

echo "=== Needle in Haystack Ablation Study ==="
echo "Model: $MODEL"
echo "Needle type: $NEEDLE_TYPE"
echo "Num needles: ${NUM_NEEDLES[*]}"
echo "Positions: ${POSITIONS[*]}"
echo "Variance: $VARIANCE"
echo "Context lengths (lines): ${NUM_LINES[*]}"
echo "Samples per config: $NUM_EXAMPLES"
echo "Metrics output: $METRICS_FILE"
echo ""

# Total configs: 6 lengths × 4 positions × 3 needle counts × 2 modes = 144 runs
total_configs=$((${#NUM_LINES[@]} * ${#POSITIONS[@]} * ${#NUM_NEEDLES[@]} * 2))
echo "Total configurations: $total_configs"
echo ""

# Remove old metrics file to start fresh (optional - comment out to append)
rm -f "$METRICS_FILE" "${METRICS_FILE%.json}.lock"

run_count=0
for num_lines in "${NUM_LINES[@]}"; do
    for pos in "${POSITIONS[@]}"; do
        for num_needles in "${NUM_NEEDLES[@]}"; do
            run_count=$((run_count + 1))
            echo "========================================"
            echo "[$run_count/$total_configs] num_lines=$num_lines, pos=$pos, needles=$num_needles"
            echo "========================================"
            
            # RLM mode
            echo "[RLM] Running..."
            uv run vf-eval needle_in_haystack \
                -m "$MODEL" \
                -n "$NUM_EXAMPLES" \
                -r "$ROLLOUTS" \
                -a "{\"num_lines\": $num_lines, \"num_needles\": $num_needles, \"needle_type\": \"$NEEDLE_TYPE\", \"needle_position\": $pos, \"needle_variance\": $VARIANCE, \"use_rlm\": true, \"seed\": $SEED, \"metrics_output_path\": \"$METRICS_FILE\"}"
            
            run_count=$((run_count + 1))
            
            # Standard LLM mode
            echo "[Standard LLM] Running..."
            uv run vf-eval needle_in_haystack \
                -m "$MODEL" \
                -n "$NUM_EXAMPLES" \
                -r "$ROLLOUTS" \
                -a "{\"num_lines\": $num_lines, \"num_needles\": $num_needles, \"needle_type\": \"$NEEDLE_TYPE\", \"needle_position\": $pos, \"needle_variance\": $VARIANCE, \"use_rlm\": false, \"seed\": $SEED, \"metrics_output_path\": \"$METRICS_FILE\"}"
            
            echo ""
        done
    done
done

echo "=== Ablation Complete ==="
echo "Results saved to: $METRICS_FILE"
echo ""
echo "Analysis example:"
echo "  import pandas as pd"
echo "  import seaborn as sns"
echo "  df = pd.read_json('$METRICS_FILE')"
echo "  # Group by num_needles and use_rlm"
echo "  df.groupby(['num_needles', 'use_rlm'])['partial_match'].mean()"


#!/bin/bash
# Run all ablation experiments sequentially
# If one crashes, continue to the next
# Usage: ./run_all_ablations.sh

# Don't use set -e - we want to continue on errors

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/ablation_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# List of environments to run
ENVIRONMENTS=(
    "math_python"
    "deepdive"
    "verbatim_copy"
    "needle_in_haystack"
    "oolong"
)

echo "=== Starting All Ablations at $(date) ==="
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Track results
declare -A RESULTS

for env in "${ENVIRONMENTS[@]}"; do
    SCRIPT="$SCRIPT_DIR/$env/run_ablations.sh"
    LOG_FILE="$LOG_DIR/${env}_${TIMESTAMP}.log"
    
    echo "############################################################"
    echo "### Starting: $env"
    echo "### Time: $(date)"
    echo "### Log: $LOG_FILE"
    echo "############################################################"
    echo ""
    
    if [[ -f "$SCRIPT" ]]; then
        # Run the script, capturing output to log file and console
        # Continue even if it fails
        if bash "$SCRIPT" 2>&1 | tee "$LOG_FILE"; then
            RESULTS[$env]="SUCCESS"
            echo ""
            echo "✓ $env completed successfully"
        else
            RESULTS[$env]="FAILED (exit code: $?)"
            echo ""
            echo "✗ $env failed - continuing to next environment"
        fi
    else
        RESULTS[$env]="SKIPPED (script not found)"
        echo "⚠ Script not found: $SCRIPT"
    fi
    
    echo ""
    echo ""
done

# Print summary
echo "############################################################"
echo "### SUMMARY"
echo "### Finished at: $(date)"
echo "############################################################"
echo ""

for env in "${ENVIRONMENTS[@]}"; do
    echo "$env: ${RESULTS[$env]}"
done

echo ""
echo "Logs saved to: $LOG_DIR"

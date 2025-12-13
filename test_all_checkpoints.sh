#!/bin/bash
# Test all checkpoints in a directory by calling single-checkpoint test repeatedly

if [ "$#" -lt 1 ]; then
    echo "Usage: ./test_all_checkpoints.sh <checkpoints_dir> [num_episodes] [num_envs]"
    echo "Example: ./test_all_checkpoints.sh results/.../checkpoints 50 300"
    exit 1
fi

CKPT_DIR="$1"
NUM_EPISODES="${2:-100}"
NUM_ENVS="${3:-300}"

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Directory not found: $CKPT_DIR"
    exit 1
fi

# Find all checkpoint subdirectories (10M, 20M, etc.)
CHECKPOINTS=$(find "$CKPT_DIR" -maxdepth 1 -type d -name "*M" | sort -V)

if [ -z "$CHECKPOINTS" ]; then
    echo "ERROR: No checkpoint directories found in $CKPT_DIR"
    exit 1
fi

echo "Found checkpoints:"
echo "$CHECKPOINTS" | while read ckpt; do echo "  - $(basename $ckpt)"; done
echo ""

# Output file
OUTPUT_FILE="$CKPT_DIR/test_results.txt"
echo "======================================================================" > "$OUTPUT_FILE"
echo "MAPush HAPPO Testing Results" >> "$OUTPUT_FILE"
echo "======================================================================" >> "$OUTPUT_FILE"
echo "Test Date: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
echo "Num Episodes per Checkpoint: $NUM_EPISODES" >> "$OUTPUT_FILE"
echo "Num Parallel Envs: $NUM_ENVS" >> "$OUTPUT_FILE"
echo "======================================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "Results by Checkpoint:" >> "$OUTPUT_FILE"
echo "----------------------------------------------------------------------" >> "$OUTPUT_FILE"

# Test each checkpoint
echo "$CHECKPOINTS" | while read CKPT_PATH; do
    CKPT_NAME=$(basename "$CKPT_PATH")

    echo "Testing checkpoint: $CKPT_NAME"

    # Run single checkpoint test and capture output
    OUTPUT=$(./run_testing.sh \
        --checkpoint "$CKPT_PATH" \
        --mode calculator \
        --num_episodes "$NUM_EPISODES" \
        --num_envs "$NUM_ENVS" 2>&1)

    # Extract success rate from output
    SUCCESS_RATE=$(echo "$OUTPUT" | grep "Success Rate:" | grep -oP '\d+\.\d+' | head -1)
    SUCCESS_PCT=$(echo "$OUTPUT" | grep "Success Rate:" | grep -oP '\(\d+\.\d+%\)' | head -1 | tr -d '()')
    NUM_SUCCESS=$(echo "$OUTPUT" | grep "episodes succeeded" | grep -oP '\[\d+' | tr -d '[')

    if [ -z "$SUCCESS_RATE" ]; then
        SUCCESS_RATE="0.0000"
        SUCCESS_PCT="0.00%"
        NUM_SUCCESS="0"
    fi

    # Append to results file
    printf "%-8s  Success Rate: %s (%s)  [%s/%s episodes]\n" \
        "$CKPT_NAME" "$SUCCESS_RATE" "$SUCCESS_PCT" "$NUM_SUCCESS" "$NUM_EPISODES" >> "$OUTPUT_FILE"

    echo "  Success Rate: $SUCCESS_RATE ($SUCCESS_PCT)"
    echo ""
done

echo "======================================================================" >> "$OUTPUT_FILE"

echo ""
echo "======================================================================"
echo "Summary of All Checkpoints"
echo "======================================================================"
cat "$OUTPUT_FILE" | grep -A 100 "Results by Checkpoint:"
echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""

#!/bin/bash
# train_loop.sh - Automated AlphaZero Training Cycle
# Usage: ./train_loop.sh [ITERATIONS] [N_PROCS] [GAMES_PER_PROC]

NUM_ITERATIONS=${1:-10}
N_PROCS=${2:-4}
GAMES_PER_PROC=${3:-100} # default: 4 * 100 = 400 games per iter

# Paths
SELFPLAY_SCRIPT="./scripts/parallel_selfplay.sh"
DATA_FILE="data/selfplay/new_data.bin"
WEIGHTS_FILE="models/cnn_weights.bin"
ARCHIVE_DIR="data/iteration"
MODEL_ARCHIVE="models/archive"

# Setup
mkdir -p "$ARCHIVE_DIR" "$MODEL_ARCHIVE" logs

echo "=== AlphaZero Loop Started ==="
echo "Iterations:  $NUM_ITERATIONS"
echo "Parallelism: $N_PROCS processes"
echo "Games/Iter:  $((N_PROCS * GAMES_PER_PROC))"

for i in $(seq 1 $NUM_ITERATIONS); do
    echo ""
    echo "=========================================="
    echo "=== ITERATION $i / $NUM_ITERATIONS ==="
    echo "=========================================="
    
    # 1. GENERATION (Self-Play)
    echo "[1/3] Generating Self-Play Data..."
    # Clean previous raw files to avoid merging old stuff
    rm -f data/selfplay/raw_gen_*.bin
    
    # Run Parallel Generation
    $SELFPLAY_SCRIPT $N_PROCS $GAMES_PER_PROC
    
    if [ ! -f "$DATA_FILE" ]; then
        echo "Error: Data generation failed. File $DATA_FILE not found."
        exit 1
    fi
    
    # Archive Data
    cp "$DATA_FILE" "${ARCHIVE_DIR}/iter_${i}_data.bin"
    echo "Saved data to ${ARCHIVE_DIR}/iter_${i}_data.bin"

    # 2. BACKUP CURRENT MODEL (Before Training)
    # The 'current' model generated the data, so we save it as the 'teacher' for this iter
    cp "$WEIGHTS_FILE" "${MODEL_ARCHIVE}/iter_${i}_weights.bin"
    echo "Backed up weights to ${MODEL_ARCHIVE}/iter_${i}_weights.bin"

    # 3. TRAINING
    echo "[2/3] Training on New Data..."
    # Launch trainer (Trainer will load existing cnn_weights.bin, train, and save back to it)
    ./bin/trainer "$DATA_FILE"
    
    echo "[3/3] Iteration $i Complete."
    echo "New weights saved to $WEIGHTS_FILE"
done

echo ""
echo "=== Training Loop Completed ==="

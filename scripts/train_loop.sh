#!/bin/bash
# scripts/train_loop.sh - Adaptive Self-Play Training Loop
# Features:
# - Mixes Standard Games (for opening/midgame) with Endgame Games (for tactics)
# - Automatically scales Batch Size and Learning Rate based on dataset size
# - Uses 'active.dat' as the replay buffer (accumulates history)

set -e

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_FILE="out/data/active.dat"
MODEL_FILE="out/models/cnn_weights.bin"
BACKUP_FILE="out/models/cnn_weights_backup.bin"

# Generation per Loop
GAMES_STANDARD=200
GAMES_ENDGAME=100
ENDGAME_PROB=0.8

# Start
echo "================================================================================"
echo "  DAMA AI - ADAPTIVE TRAINING LOOP"
echo "================================================================================"
echo "Data File:  $DATA_FILE"
echo "Model File: $MODEL_FILE"
echo ""

mkdir -p out/data out/models out/logs

# Ensure initial model exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: $MODEL_FILE missing. Please initialize or copy v3/weights."
    exit 1
fi

LOOP_COUNT=1

while true; do
    echo "--------------------------------------------------------------------------------"
    echo "  LOOP #$LOOP_COUNT - $(date)"
    echo "--------------------------------------------------------------------------------"

    # 1. SELF-PLAY (STANDARD)
    # -----------------------
    echo "> Generating Standard Games ($GAMES_STANDARD)..."
    ./bin/dama train \
        --selfplay \
        --games $GAMES_STANDARD \
        --endgame-prob 0.05 \
        --temp 1.0 \
        --data "$DATA_FILE" \
        --weights "$MODEL_FILE" \
        --workers 8 \
        | tee -a out/logs/selfplay_std.log

    # 2. SELF-PLAY (ENDGAME FOCUSED)
    # ------------------------------
    echo "> Generating Endgame Scenarios ($GAMES_ENDGAME)..."
    ./bin/dama train \
        --selfplay \
        --games $GAMES_ENDGAME \
        --endgame-prob $ENDGAME_PROB \
        --temp 1.0 \
        --data "$DATA_FILE" \
        --weights "$MODEL_FILE" \
        --workers 8 \
        | tee -a out/logs/selfplay_end.log

    # 3. ANALYZE DATASET SIZE
    # -----------------------
    # Extract sample count from inspect command
    SAMPLE_COUNT=$(./bin/dama data inspect "$DATA_FILE" | grep "Samples:" | head -1 | tr -d ',' | awk '{print $2}')
    if [ -z "$SAMPLE_COUNT" ]; then SAMPLE_COUNT=0; fi
    
    echo "> Dataset Size: $SAMPLE_COUNT samples"

    # 4. COMPUTE HYPERPARAMETERS
    # --------------------------
    # Adaptive Logic:
    # < 500k:   Batch 64/128, LR 0.5 (Base Policy - High Plasticity)
    # > 500k:   Batch 256,    LR 0.2 (Stability)
    
    BATCH_SIZE=64
    LR=0.5
    EPOCHS=5 

    if [ "$SAMPLE_COUNT" -gt 500000 ]; then
        BATCH_SIZE=256
        LR=0.2
        EPOCHS=3 
    elif [ "$SAMPLE_COUNT" -gt 100000 ]; then
        BATCH_SIZE=128
        LR=0.5
        EPOCHS=3
    fi

    echo "> Adaptive Config: Batch=$BATCH_SIZE | LR=$LR | Epochs=$EPOCHS"

    # 5. TRAINING
    # -----------
    echo "> Training..."
    ./bin/dama train \
        --train \
        --data "$DATA_FILE" \
        --weights "$MODEL_FILE" \
        --epochs $EPOCHS \
        --batch $BATCH_SIZE \
        --lr $LR \
        | tee -a out/logs/training.log

    # 6. CLEANUP / ROTATION (Optional)
    # --------------------------------
    # (Here we could run tidy.sh if we wanted to archive active.dat, 
    # but we want to accumulate for now as Replay Buffer).

    # Increment
    LOOP_COUNT=$((LOOP_COUNT+1))
    
    # Brief pause to let system cool / user interrupt
    sleep 2
done

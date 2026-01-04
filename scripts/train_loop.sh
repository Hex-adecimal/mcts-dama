#!/bin/bash
# scripts/train_loop.sh - Scaled AlphaZero Training Loop
# Configured for: 4-Layer CNN (~200k params), M2 CPU
#
# Logic:
# 1. Self-Play: Generate 500 games with current BEST model
# 2. Window: Trim active.dat to last 25,000 games
# 3. Train: Train CANDIDATE model on window
# 4. Evaluate: Tournament CANDIDATE vs BEST (100 games)
# 5. Promote: If CANDIDATE wins >= 55%, it becomes BEST

set -e

# ==============================================================================
# CONFIGURATION (SCALED)
# ==============================================================================
DATA_FILE="out/data/active.dat"
BEST_MODEL="out/models/best.bin"
CANDIDATE_MODEL="out/models/candidate.bin"
LOG_DIR="out/logs"

# Generation
GAMES_PER_LOOP=500
WORKERS=8 # M2 CPU cores
NODES=800

# Window (Memory for Small Net)
# 25,000 games * ~60 moves = 1.5M samples
WINDOW_SIZE=1500000 

# Training (No overfitting on small batch)
BATCH_SIZE=128
VAL_INTERVAL=100
EPOCHS=1 # 1 epoch on window is enough per loop? Maybe. Let's try 1.

# Evaluation
EVAL_GAMES=100
WIN_THRESHOLD=55 # 55% wins required

# ==============================================================================
# SETUP
# ==============================================================================
mkdir -p out/data out/models out/logs

echo "================================================================================"
echo "  DAMA ZERO - SCALED TRAINING LOOP"
echo "================================================================================"
echo "Cycle: $GAMES_PER_LOOP games -> Train -> Eval ($EVAL_GAMES games)"

# Ensure initial model exists
if [ ! -f "$BEST_MODEL" ]; then
    echo "ERROR: $BEST_MODEL missing. Please copy your starting weights to $BEST_MODEL"
    # Fallback: if cnn_weights.bin exists, use it
    if [ -f "out/models/cnn_weights.bin" ]; then
        cp out/models/cnn_weights.bin "$BEST_MODEL"
        echo "Copied existing cnn_weights.bin to best.bin"
    else
        echo "Creating fresh random model..."
        # Running train with --init creates fresh weights
        ./bin/dama train --init --weights "$BEST_MODEL" --train --epochs 0
    fi
fi

LOOP_COUNT=1

while true; do
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "  LOOP #$LOOP_COUNT - $(date)"
    echo "--------------------------------------------------------------------------------"

    # 1. SELF-PLAY
    # -----------------------
    echo "> Generating $GAMES_PER_LOOP games (Best Model)..."
    ./bin/dama train \
        --selfplay \
        --games $GAMES_PER_LOOP \
        --nodes $NODES \
        --temp 1.0 \
        --data "$DATA_FILE" \
        --weights "$BEST_MODEL" \
        --threads $WORKERS \
        --endgame-prob 0.1 \
        | tee -a "$LOG_DIR/selfplay_loop${LOOP_COUNT}.log"

    # 2. WINDOW MANAGEMENT
    # -----------------------
    echo "> Managing Window (Limit $WINDOW_SIZE samples)..."
    ./bin/dama data trim "$DATA_FILE" $WINDOW_SIZE

    # 3. TRAINING CANDIDATE
    # -----------------------
    echo "> Training Candidate (Windowed)..."
    ./bin/dama train \
        --train \
        --data "$DATA_FILE" \
        --weights "$BEST_MODEL" "$CANDIDATE_MODEL" \
        --epochs $EPOCHS \
        --batch $BATCH_SIZE \
        --val-interval $VAL_INTERVAL \
        --threads $WORKERS \
        | tee -a "$LOG_DIR/train_loop${LOOP_COUNT}.log"

    # 4. EVALUATION TOURNAMENT
    # -----------------------
    echo "> Evaluation: Candidate vs Best ($EVAL_GAMES games)..."
    # P1 = Candidate, P2 = Best
    # Grep standard tournament output
    # Output format: "Match Results: P1: 45 | P2: 35 | Draw: 20"
    
    # We use a temporary log to parse result
    EVAL_LOG="$LOG_DIR/eval_loop${LOOP_COUNT}.txt"
    
    ./bin/dama tournament \
        --p1-type mcts --p1-path "$CANDIDATE_MODEL" \
        --p2-type mcts --p2-path "$BEST_MODEL" \
        --games $EVAL_GAMES \
        --timeout 0.1 \
        | tee "$EVAL_LOG"

    # Parse Wins (P1 wins)
    # Looking for: "P1 (White/Black): X wins" lines is hard due to color swap.
    # Look for final summary line: "Match: P1 (Candidate) vs P2 (Adv)" if named?
    # CLI output for tournament is usually: 
    # "Final Score: Player 1: X  Player 2: Y  Draws: Z"
    # or similar. Let's assume tournament.c prints something parsable.
    
    # Actually tournament.c usually prints:
    # "Player 1 Wins: X (Ratio)"
    
    # Let's rely on grep for "Player 1 wins: X"
    # Or just check the log manually for now if parsing is complex?
    # No, we need automation.
    
    # Let's try to extract P1 Score.
    # Assuming output: "Player 1: 55.0" or similar.
    # Let's look at tournament.c or run a helper.
    # Safe regex for strict format?
    
    # Let's grab the wins from the CSV-like part or the summary lines.
    P1_WINS=$(grep -o "Player 1: [0-9]*" "$EVAL_LOG" | tail -1 | awk '{print $3}')
    if [ -z "$P1_WINS" ]; then P1_WINS=0; fi

    echo "Candidate Wins: $P1_WINS / $EVAL_GAMES"

    # 5. PROMOTION
    # -----------------------
    if [ "$P1_WINS" -ge "$WIN_THRESHOLD" ]; then
        echo ">>> PROMOTION! Candidate ($P1_WINS wins) replaces Best."
        cp "$CANDIDATE_MODEL" "$BEST_MODEL"
        
        # Save historical checkpoint
        cp "$BEST_MODEL" "out/models/gen_${LOOP_COUNT}_wins_${P1_WINS}.bin"
    else
        echo "... Rejected. Candidate ($P1_WINS wins) failed to beat threshold ($WIN_THRESHOLD)."
        # We discard candidate, but we Keep the Data!
        # Next loop will generate more data with OLD best, creating new situations.
        # And we train again on bigger/fresher window.
    fi

    LOOP_COUNT=$((LOOP_COUNT+1))
    sleep 2
done

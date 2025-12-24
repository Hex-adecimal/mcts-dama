#!/bin/bash
# Usage: ./scripts/parallel_selfplay.sh [N_PROCESSES] [GAMES_PER_PROCESS]

N_PROCS=${1:-4}        # Default 4 processes (Safe for M2 usage)
GAMES_PER_PROC=${2:-50} # Default 50 games each -> 200 games total (Fast check)

echo "=== Starting Parallel Self-Play ==="
echo "Processes:     $N_PROCS"
echo "Games/Proc:    $GAMES_PER_PROC"
echo "Total Games:   $((N_PROCS * GAMES_PER_PROC))"
echo "Output Dir:    data/selfplay/"

mkdir -p data/selfplay
# Optional: Clear old runs? No, let's keep them or manual clear.
# rm -f data/selfplay/raw_gen_*.bin

# Ensure binaries are ready
make selfplay merger > /dev/null

pids=""
for i in $(seq 1 $N_PROCS); do
    # Run in background. Output suppressed or redirected if needed.
    ./bin/selfplay $GAMES_PER_PROC $i > /dev/null &
    pids="$pids $!"
done

echo "Running with PIDs: $pids"
wait $pids

echo "Self-play complete. Merging..."
# Merge all generated files into one
./bin/merger data/selfplay/new_data.bin data/selfplay/raw_gen_*.bin

echo "Merged to data/selfplay/new_data.bin"
ls -lh data/selfplay/new_data.bin

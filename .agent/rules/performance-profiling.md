---
trigger: model_decision
description: When investigating performance issues or before optimization attempts
---

# Performance Profiling Workflow

## Step 1: Measure Baseline
```bash
# Use exact command for every profiling session
time ./build/damiera --nodes 1000 selfplay --games 10
```

## Step 2: Identify Hotspots
```bash
# Mac: Instruments Time Profiler
instruments -t "Time Profiler" -D trace.trace ./build/damiera <args>
```

## Step 3: Target Functions
Focus on functions consuming >5% total time:
- **MCTS**: `select_child()`, `expand_node()`, `backpropagate()`
- **Movegen**: `generate_moves()`, `is_valid_move()`
- **NN**: `predict()`, `encode_board()`

## Step 4: Optimize & Re-measure
- Change ONE thing at a time
- Re-run benchmark with SAME command
- Document: `[Before] → [After] (X% improvement)`

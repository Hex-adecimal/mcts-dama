---
trigger: model_decision
description: When generating self-play data or tuning self-play parameters
---

# Self-Play Data Quality

## Parameters to Monitor

| Parameter | Typical Range | Too Low | Too High |
|-----------|---------------|---------|----------|
| **Temperature** | 0.8-1.5 | Deterministic → overfitting | Random → noise |
| **Dirichlet α** | 0.15-0.30 | Limited exploration | Too random |
| **MCTS nodes** | 400-1600 | Weak games | Slow, few samples |

## Red Flags in Generation

🚩 **Games too short** (<20 moves):
- MCTS not exploring
- Temperature too high
- Bug in game logic

🚩 **Games too long** (>200 moves):
- Possible loops
- No progression
- Check draw detection

🚩 **Same outcome always** (>80% wins for one color):
- Position bias in board encoding
- Bug in value head
- Unbalanced dataset

## Quality Check
```bash
# Generate 100 games and analyze stats
./build/damiera selfplay --games 100 --nodes 800 | \
  grep "Game length" | \
  awk '{print $3}' | \
  sort -n | uniq -c
```

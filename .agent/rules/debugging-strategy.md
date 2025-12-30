---
trigger: model_decision
description: When debugging issues or unexpected behavior
---

# Debugging Strategy

Follow this order:

1. **Reproduce**: Create minimal case with fixed seed
2. **Check basics**: Verify movegen → eval → MCTS
3. **Structured logging**: Use DEBUG/INFO/WARN/ERROR levels
4. **Visualize MCTS tree**: Print structure to verify exploration
5. **Compare baseline**: Check behavior vs MCTS vanilla
6. **Use asserts**: Add invariant checks (e.g., valid moves)

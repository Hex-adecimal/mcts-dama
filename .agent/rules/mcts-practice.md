---
trigger: model_decision
description: When implementing or modifying MCTS-related code
---

# MCTS Best Practices

1. **UCB1 tuning**: C_puct typically 1.0-2.0, needs empirical tuning
2. **Transposition table**: Consider hash table for visited states
3. **Early termination**: Stop search if move is clearly winning
4. **Memory management**: Use object pooling for tree nodes in C
5. **Parallelization**: Virtual loss for parallel MCTS, watch race conditions

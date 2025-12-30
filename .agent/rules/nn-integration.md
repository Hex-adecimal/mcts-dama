---
trigger: model_decision
description: When working on neural network code or training pipeline
---

# NN Integration (AlphaZero-style)

1. **Input encoding**: Always normalize values (0-1 or -1 to 1)
2. **Policy head**: Softmax on legal moves, mask illegal moves
3. **Value head**: Single output with tanh ∈ [-1, 1]
4. **Batch inference**: Accumulate states for batch prediction
5. **Training data**: Balance wins/losses/draws in dataset
6. **Loss function**: Combine cross-entropy (policy) + MSE (value)

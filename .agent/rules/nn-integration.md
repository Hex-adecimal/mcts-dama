---
trigger: model_decision
description: When working on neural network code or training pipeline
---

Per l'integrazione NN stile AlphaZero:

1. **Input encoding**: normalizza sempre i valori (0-1 o -1 a 1)
2. **Policy head**: usa softmax su mosse legali, maschera le mosse illegali
3. **Value head**: output singolo con tanh ∈ [-1, 1]
4. **Batch inference**: accumula stati per batch prediction quando possibile
5. **Training data**: bilancia vittorie/sconfitte/pareggi nel dataset
6. **Loss function**: combina cross-entropy (policy) + MSE (value)

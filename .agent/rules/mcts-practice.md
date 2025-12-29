---
trigger: model_decision
description: When implementing or modifying MCTS-related code
---

Segui queste best practices per MCTS:

1. **UCB1 tuning**: C_puct tipicamente tra 1.0-2.0, richiede tuning empirico
2. **Transposition table**: considera una hash table per stati già visitati
3. **Early termination**: interrompi la ricerca se una mossa è chiaramente vincente
4. **Memory management**: usa object pooling per i nodi dell'albero in C
5. **Parallelizzazione**: virtual loss per MCTS parallelo, attenzione a race conditions

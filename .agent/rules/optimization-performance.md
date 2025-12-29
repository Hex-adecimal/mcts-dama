---
trigger: model_decision
description: When optimizing code or discussing performance bottlenecks
---

Prima di ottimizzare, profilare sempre il codice:

1. Usa `time` o strumenti come `perf`/`Instruments` per identificare le funzioni lente
2. Concentrati sulle hot-path: loop interni di MCTS, generazione mosse, valutazione NN
3. Misura PRIMA e DOPO ogni ottimizzazione con benchmark riproducibili
4. Preferisci ottimizzazioni algoritmiche (O(n²) → O(n)) rispetto a micro-ottimizzazioni

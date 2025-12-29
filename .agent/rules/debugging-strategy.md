---
trigger: model_decision
description: When debugging issues or unexpected behavior
---

Segui questo ordine per il debugging:

1. **Riproduci il bug**: crea un caso minimo riproducibile con seed fisso
2. **Verifica le basi**: controlla prima movegen, poi valutazione, poi MCTS
3. **Logging strutturato**: aggiungi log con livelli (DEBUG/INFO/WARN/ERROR)
4. **Visualizza l'albero MCTS**: stampa la struttura per verificare l'esplorazione
5. **Confronta con baseline**: verifica il comportamento vs MCTS vanilla
6. **Usa assert**: aggiungi assert per invarianti (es: mosse valide)

# 🚀 Parallelization & Optimization Report

Questo documento riassume tutte le strategie di parallelizzazione e ottimizzazione implementate nel progetto **MCTS Dama**, indicando come vengono sfruttate le risorse hardware (CPU multi-core, Apple Silicon M2). Infine, suggerisce ulteriori punti di miglioramento.

---

## 🧠 1. CNN (Convolutional Neural Network)

La CNN è il cuore computazionale ed è stata pesantemente ottimizzata per l'architettura Apple M2.

### ✅ Implementato

| Componente | Tecnica | Descrizione | File |
|------------|---------|-------------|------|
| **Training Loop** | `OpenMP` | Parallelizzazione data-parallel sui batch di training per calcolare gradienti parziali. | `cnn_training.c` |
| **Validation Loop** | `OpenMP Redux` | Calcolo parallelo della loss su batch di validazione con riduzione (`reduction`) sicura. | `training_pipeline.c` |
| **Convoluzioni** | `Accelerate` | Uso di `cblas_sgemm` (Matrix-Matrix mult) per ottimizzare i filtri convoluzionali. | `conv_ops.c` |
| **Fully Connected** | `Accelerate` | Uso di `cblas_sgemv` (Matrix-Vector) per forward pass e `cblas_sger` (Outer Product) per backward pass. | `cnn_training.c` |
| **Softmax** | `vDSP` | Vettorizzazione SIMD Apple-specific (`vvexpf`, `vDSP_sve`) per calcolare l'esponenziale e la somma velocemente. | `cnn_inference.c` |
| **Batch Norm** | `OpenMP` | Parallelismo sui canali per calcolo media/varianza e normalizzazione. | `cnn_batch_norm.c` |
| **Update Pesi** | `OpenMP` | Aggiornamento parallelo dei pesi (SGD + Momentum) per ogni parametro. | `cnn_training.c` |

---

## 🌳 2. MCTS & Tournament

L'ambiente di gioco e valutazione sfrutta parallelismo massiccio.

### ✅ Implementato

| Componente | Tecnica | Descrizione | File |
|------------|---------|-------------|------|
| **Async Batching** | `Pthreads` | I worker thread (esploratori dell'albero) non eseguono la rete neurale direttamente. Invece, accodano richieste in una `InferenceQueue`. Un thread \"Master\" processa un batch intero (es. 256 stati) in parallelo usa CNN ottimizzata. | `mcts_search.c`, `mcts_worker.c` |
| **Thread Safety** | `Mutex/Cond` | Protezione delle sezioni critiche dell'albero e della coda di inferenza. | `mcts_tree.c`, `mcts_worker.c` |
| **Virtual Loss** | `Atomics` | Penalizza temporaneamente i nodi in esplorazione per prevenire thread collision. | `mcts_selection.c` |
| **Tournament** | `OpenMP` | **Root Parallelization (Match Level):** Le partite di un torneo vengono giocate in parallelo su thread CPU separati (`#pragma omp parallel for`), garantendo scaling lineare. | `tournament.c` |

---

## 🔮 3. Future Parallelization Opportunities

Aree dove è possibile ottenere ulteriore speedup.

### ⚡️ Micro-Optimizations

1. **Optimized Move Generation (Bitboards):**
   - **Stato attuale:** `generate_moves` usa loop e logica ricorsiva per le catene di cattura. Usa bitboards per check semplici ma non per generazione massiva parallela (SIMD).
   - **Benefit:** Su M2 (ARM64), l'uso di intrinsics NEON per generare mosse legali per tutte le pedine in parallelo darebbe uno speedup significativo alle simulazioni "vanilla".

2. **Half-Precision (FP16):**
   - **Stato attuale:** Tutto `float` (FP32).
   - **Benefit:** I core AMX di M2 sono ottimizzati per matrici FP16/BF16. Convertire l'inferenza a FP16 potrebbe raddoppiare il throughput teorico della CNN.

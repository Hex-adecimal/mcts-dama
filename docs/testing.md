# Testing e Benchmarking Reference

Documentazione del sistema di testing e benchmarking per MCTS Dama, con focus su metodologia, coverage e risultati.

---

## 1. Quick Start

```bash
make test    # Esegue tutti i 79 unit test (~2 secondi)
make bench   # Esegue tutti i benchmark (~30 secondi)
```

---

## 2. Metodologia di Testing

### Filosofia

Il sistema di test segue i principi:

1. **Unit Testing**: Ogni funzione critica ha test isolati
2. **Property-Based**: Alcuni test verificano invarianti (es. "policy sum ≈ 1.0")
3. **Regression**: Test specifici per bug risolti
4. **Determinismo**: Test ripetibili con RNG seedato

### Framework Custom

Il framework di testing è implementato in C puro (nessuna dipendenza esterna):

```c
// test_framework.h
#define ASSERT_EQ(a, b)           // Confronto esatto
#define ASSERT_FLOAT_EQ(a, b, e)  // Confronto con epsilon
#define ASSERT_TRUE(x)            // Boolean check
#define ASSERT_NOT_NULL(p)        // Null safety
```

---

## 3. Unit Tests (79 test)

### Struttura

```
tests/unit/
├── test_main.c         # Runner principale
├── test_framework.h    # Macro ASSERT_*
├── test_engine.c       # 16 test
├── test_search.c       # 21 test
├── test_neural.c       # 12 test
├── test_training.c     # 10 test
└── test_common.c       # 20 test
```

### Coverage per Modulo

| Modulo | Test | Coverage Funzionale |
|--------|------|---------------------|
| **Engine** | 16 | init_game, movegen, catture, promozione, zobrist, bit ops |
| **Search** | 21 | arena, presets, UCB, PUCT, backprop, solver, TT, tree reuse |
| **Neural** | 12 | init, forward, batch, encode, save/load, determinismo |
| **Training** | 10 | dataset I/O, shuffle, split, train_step, gradienti |
| **Common** | 20 | RNG (seed, u32, f32, gamma), params, error codes, debug assertions |

### Test Significativi per Categoria

#### Engine (Correttezza Regole)

| Test | Verifica |
|------|----------|
| `engine_captures_are_mandatory` | Regola cattura obbligatoria |
| `engine_promotion_to_lady` | Promozione corretta a dama |
| `engine_zobrist_keys_are_unique` | Unicità hash per TT |
| `engine_movegen_generate_initial_position_has_7_moves` | 7 mosse legali da posizione iniziale |

#### Search (Algoritmi MCTS)

| Test | Verifica |
|------|----------|
| `search_ucb1_score_increases_with_visits` | Formula UCB1 corretta |
| `search_puct_uses_priors` | PUCT integra prior CNN |
| `search_backprop_updates_scores` | Backpropagation corretto |
| `search_virtual_loss_zeroed_after_search` | Virtual loss azzerato dopo ricerca |
| `search_solver_detects_terminal` | Solver identifica stati terminali |
| `search_tree_reuse_preserves_stats` | Statistiche preservate in tree reuse |

#### Neural (Rete Neurale)

| Test | Verifica |
|------|----------|
| `neural_cnn_forward_is_deterministic` | Stesso input → stesso output |
| `neural_cnn_forward_produces_valid_output` | Value ∈ [-1,1], policy ≥ 0 |
| `neural_cnn_save_and_load_roundtrip` | Persistenza pesi funziona |
| `neural_move_to_index_different_for_colors` | Encoding canonico per colore |

#### Training (Pipeline)

| Test | Verifica |
|------|----------|
| `training_cnn_train_step_reduces_loss` | Loss decresce con training |
| `training_cnn_gradients_cleared_after_update` | Gradienti azzerati dopo update |
| `training_sample_policy_sums_to_valid` | Policy target normalizzata |

#### Common (Utilities)

| Test | Verifica |
|------|----------|
| `common_rng_gamma_mean_approximately_alpha` | RNG gamma ha media corretta |
| `common_rng_u32_is_deterministic` | RNG ripetibile con stesso seed |
| `common_popcount_works` | Bit counting corretto |

### Esecuzione Filtrata

```bash
./bin/run_tests engine    # Solo engine tests
./bin/run_tests search    # Solo search tests  
./bin/run_tests neural    # Solo neural tests
./bin/run_tests training  # Solo training tests
./bin/run_tests common    # Solo common tests
```

---

## 4. Benchmark (20+ metriche)

### Struttura

```
tests/benchmark/
├── bench_framework.h   # Utilities timing
└── bench_main.c        # Runner principale (~500 righe)
```

### Risultati Benchmark (Apple M2, Gennaio 2026)

#### Engine Module

| Operazione | Throughput | Latenza | Significato |
|------------|------------|---------|-------------|
| `movegen: initial` | **9.95M ops/sec** | 0.10 μs | Velocità generazione mosse |
| `movegen: midgame` | **10.75M ops/sec** | 0.09 μs | Performance con più pezzi |
| `apply_move` | **59.79M ops/sec** | 0.02 μs | Aggiornamento stato O(1) |
| `init_game + zobrist` | **66.61M ops/sec** | 0.02 μs | Setup posizione |
| `zobrist_compute_hash` | **62.98M ops/sec** | 0.02 μs | Full hash recompute |
| `endgame: random` | **6.16M ops/sec** | 0.16 μs | Setup posizione endgame |

#### Neural Module

| Operazione | Throughput | Latenza | Significato |
|------------|------------|---------|-------------|
| `cnn_forward: single` | **1,568 ops/sec** | 638 μs | Latenza singola valutazione |
| `cnn_forward_with_history` | **1,641 ops/sec** | 609 μs | Con encoding history |
| `cnn_forward_batch: 16` | **109 batches/sec** | 9.19 ms | Throughput batch ottimizzato |
| `cnn_encode_sample` | **3.57M ops/sec** | 0.28 μs | GameState → tensor |
| `cnn_move_to_index` | **65.6M ops/sec** | 0.02 μs | Move → policy index |

#### MCTS Module

| Operazione | Throughput | Latenza | Significato |
|------------|------------|---------|-------------|
| `mcts_create_root` | **44.8K ops/sec** | 22.3 μs | Setup albero |
| `mcts: 100 nodes (Vanilla)` | **1,040 ops/sec** | 962 μs | Baseline senza CNN |
| `mcts: 500 nodes (Vanilla)` | **209 ops/sec** | 4.79 ms | Scaling lineare |
| `mcts: 100 nodes (Grandmaster)` | **1,327 ops/sec** | 754 μs | Heuristic eval |
| `mcts: 1000 nodes (AlphaZero+CNN)` | **~1 ops/sec** | 1.26 sec | Bottleneck CNN |
| `arena_alloc: 1000 nodes` | **26.4K ops/sec** | 37.9 μs | Allocazione bulk |
| `tt: create+free (4096)` | **15.2K ops/sec** | 65.9 μs | TT setup/teardown |

#### Training Module

| Operazione | Throughput | Latenza | Significato |
|------------|------------|---------|-------------|
| `cnn_train_step: batch=1` | **46 steps/sec** | 21.9 ms | Single sample |
| `cnn_train_step: batch=32` | **19 batches/sec** | 53.4 ms | 608 samples/sec |
| `dataset_shuffle: 1000` | **7,863 ops/sec** | 127 μs | Fisher-Yates |

---

## 5. Analisi Prestazionale

### Batch Efficiency CNN

```
Throughput vs Batch Size:
Batch 1:   46 samples/sec   (baseline)
Batch 16:  1,744 samples/sec (38x speedup)
Batch 32:  608 samples/sec   (13x, memory limited)
```

**Insight**: Batch size ottimale è ~16 per questo hardware (M2 8GB).

### MCTS Scaling

```
Nodes vs Time (Vanilla preset):
100 nodes:  ~1ms   → 1,040 searches/sec
500 nodes:  ~5ms   → 209 searches/sec
1000 nodes: ~10ms  → ~100 searches/sec
```

**Insight**: Scaling quasi-lineare. Overhead fisso ~200μs per search setup.

### CNN Dominance in AlphaZero

```
MCTS 1000 nodes breakdown (stima):
- Tree operations: ~10ms (1%)
- CNN inference:   ~1250ms (99%)
```

**Insight**: In modalità AlphaZero, la CNN è il bottleneck assoluto.

---

## 6. Come Eseguire

### Test Completi

```bash
make test
# Output: Passed: 79 | Failed: 0 | Total: 79
```

### Benchmark Completi

```bash
make bench
# Output: Tabella formattata con tutti i risultati
```

### Benchmark Filtrati

```bash
./bin/run_bench engine    # Solo engine benchmarks
./bin/run_bench neural    # Solo neural benchmarks
./bin/run_bench mcts      # Solo MCTS benchmarks
./bin/run_bench training  # Solo training benchmarks
```

---

## 7. Aggiungere Nuovi Test

### 1. Definire il test

```c
// In test_<module>.c
TEST(module_test_name) {
    // Setup
    GameState state;
    init_game(&state);
    
    // Action
    MoveList moves;
    movegen_generate(&state, &moves);
    
    // Assert
    ASSERT_EQ(7, moves.count);  // 7 mosse iniziali
}
```

### 2. Registrare il test

```c
// In test_main.c, dentro register_all_tests()
REGISTER_TEST(module_test_name);
```

### Macro Assert Disponibili

| Macro | Uso |
|-------|-----|
| `ASSERT_EQ(a, b)` | `a == b` |
| `ASSERT_NE(a, b)` | `a != b` |
| `ASSERT_GT(a, b)` | `a > b` |
| `ASSERT_GE(a, b)` | `a >= b` |
| `ASSERT_LT(a, b)` | `a < b` |
| `ASSERT_LE(a, b)` | `a <= b` |
| `ASSERT_TRUE(x)` | `x` is truthy |
| `ASSERT_FALSE(x)` | `x` is falsy |
| `ASSERT_NULL(x)` | `x == NULL` |
| `ASSERT_NOT_NULL(x)` | `x != NULL` |
| `ASSERT_FLOAT_EQ(a, b, eps)` | `|a-b| < eps` |

---

## 8. CI Integration

```bash
#!/bin/bash
# ci.sh - Continuous Integration script

set -e

echo "=== Building ==="
make clean && make all

echo "=== Running Tests ==="
make test
if [ $? -ne 0 ]; then
    echo "❌ Tests failed!"
    exit 1
fi

echo "=== Running Benchmarks ==="
make bench | tee benchmark_results.txt

echo "✅ All checks passed!"
```

---

## 9. Interpretazione Risultati per Presentazione

### Cosa Dimostra la Test Suite

| Aspetto | Evidenza |
|---------|----------|
| **Correttezza algoritmica** | 21 test MCTS verificano UCB/PUCT/backprop |
| **Rispetto regole** | 16 test engine verificano catture, promozioni |
| **Stabilità numerica** | Test determinismo CNN, gradienti azzerati |
| **Memory safety** | Test arena allocation, TT lifecycle |
| **Riproducibilità** | RNG seedato e verificato deterministico |

### Metriche Chiave per il Prof

1. **10.75M movegen/sec** → Engine bitboard altamente ottimizzato
2. **38x batch speedup** → BLAS ben sfruttato
3. **99% tempo in CNN** → Conferma necessità GPU per scaling
4. **79 test, 0 failures** → Codebase stabile e verificata

---

## 10. Link ai File

| File | Descrizione |
|------|-------------|
| [test_main.c](file:///Users/luigipenza/Desktop/%5B%20Intelligent%20Web%20%5D/MCTS%20Dama/tests/unit/test_main.c) | Runner e registrazione test |
| [test_framework.h](file:///Users/luigipenza/Desktop/%5B%20Intelligent%20Web%20%5D/MCTS%20Dama/tests/unit/test_framework.h) | Macro ASSERT_* |
| [bench_main.c](file:///Users/luigipenza/Desktop/%5B%20Intelligent%20Web%20%5D/MCTS%20Dama/tests/benchmark/bench_main.c) | Runner benchmark |

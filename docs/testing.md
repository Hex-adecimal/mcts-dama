# Testing e Benchmarking

Documentazione dei sistemi di test e benchmark per MCTS Dama.

---

## Quick Start

```bash
make test    # Esegue tutti i 75 unit test
make bench   # Esegue tutti i benchmark (~30 secondi)
```

---

## Unit Tests (75 test)

### Struttura

```
tests/unit/
├── test_main.c         # Runner principale
├── test_framework.h    # Macro ASSERT_*
├── test_engine.c       # 16 test
├── test_search.c       # 21 test  ← Aggiornato
├── test_neural.c       # 12 test
├── test_training.c     # 10 test
└── test_common.c       # 16 test
```

### Test per Modulo

| Modulo | Test | Copertura |
|--------|------|-----------|
| **Engine** | 16 | init_game, movegen, captures, promotion, zobrist |
| **Search** | 21 | arena, presets, root, UCB, PUCT, backprop, solver, TT |
| **Neural** | 12 | cnn_init, forward, batch, encode, save/load |
| **Training** | 10 | dataset I/O, shuffle, split, train_step |
| **Common** | 16 | RNG, params, bit operations |

### Nuovi Test Aggiunti (v2)

| Test | Descrizione |
|------|-------------|
| `search_ucb1_score_increases_with_visits` | Verifica UCB selection |
| `search_puct_uses_priors` | Verifica PUCT con prior CNN |
| `search_backprop_updates_scores` | Backpropagation aggiorna score/visits |
| `search_virtual_loss_zeroed_after_search` | Virtual loss ≤0 dopo search |
| `search_solver_detects_terminal` | Solver rileva nodi terminali |
| `search_tt_create_and_free` | TT allocation/deallocation |
| `search_tt_mask_is_power_of_two` | TT mask corretto per hashing |
| `search_tree_reuse_preserves_stats` | Tree reuse conserva statistiche |

### Esecuzione Filtrata

```bash
./bin/run_tests engine    # Solo engine tests
./bin/run_tests search    # Solo search tests  
./bin/run_tests neural    # Solo neural tests
./bin/run_tests training  # Solo training tests
./bin/run_tests common    # Solo common tests
```

---

## Benchmarks

### Struttura

```
tests/benchmark/
├── bench_framework.h   # Utilities timing
└── bench_main.c        # Runner principale (~500 righe)
```

### Benchmark Disponibili

| Categoria | Benchmark | Target |
|-----------|-----------|--------|
| **Engine** | movegen: initial | 9.7M ops/sec |
| | movegen: midgame | 10.4M ops/sec |
| | apply_move | 62M ops/sec |
| | init_game + zobrist | 66M ops/sec |
| | endgame: random | 6M ops/sec |
| **Neural** | cnn_forward: single | 1.6K ops/sec |
| | cnn_forward_with_history | 1.6K ops/sec |
| | cnn_forward_batch: 16 | 106 ops/sec |
| | cnn_encode_sample | 3.5M ops/sec |
| | cnn_move_to_index | 61M ops/sec |
| **MCTS** | mcts_create_root | ~10K ops/sec |
| | mcts: 100 nodes (Vanilla) | ~100 ops/sec |
| | mcts: 500 nodes (Vanilla) | ~20 ops/sec |
| | mcts: 100 nodes (Grandmaster) | ~50 ops/sec |
| | mcts: 1000 nodes (AlphaZero+CNN) | ~5 ops/sec |
| | arena_alloc: 1000 nodes | ~500 ops/sec |
| | tt: create+free (4096) | ~1K ops/sec |
| **Training** | cnn_train_step: batch=1 | ~400 ops/sec |
| | cnn_train_step: batch=32 | ~15 ops/sec |
| | dataset_shuffle: 1000 | ~1K ops/sec |

### Esecuzione Filtrata

```bash
./bin/run_bench engine    # Solo engine benchmarks
./bin/run_bench neural    # Solo neural benchmarks
./bin/run_bench mcts      # Solo MCTS benchmarks
./bin/run_bench training  # Solo training benchmarks
```

---

## Performance Reference (Apple M2)

| Operazione | Throughput | Note |
|------------|------------|------|
| Move generation | **10M moves/sec** | Lookup tables + bitboards |
| CNN forward (single) | **1,600/sec** | ~0.6ms latency |
| CNN forward (batch 16) | **1,700/sec** | 9.5ms per batch |
| MCTS 500 nodes | **20/sec** | ~50ms per move |
| Training step (batch 32) | **15/sec** | ~65ms per batch |

---

## Aggiungere Nuovi Test

### 1. Definire il test in `test_<module>.c`

```c
TEST(module_test_name) {
    // Setup
    GameState state;
    init_game(&state);
    
    // Action
    // ...
    
    // Assert
    ASSERT_EQ(expected, actual);
    ASSERT_NOT_NULL(ptr);
    ASSERT_TRUE(condition);
    
    // Cleanup
}
```

### 2. Registrare in `test_main.c`

```c
REGISTER_TEST(module_test_name);
```

### Macro Disponibili

| Macro | Descrizione |
|-------|-------------|
| `ASSERT_EQ(a, b)` | a == b |
| `ASSERT_NE(a, b)` | a != b |
| `ASSERT_GT(a, b)` | a > b |
| `ASSERT_GE(a, b)` | a >= b |
| `ASSERT_LT(a, b)` | a < b |
| `ASSERT_LE(a, b)` | a <= b |
| `ASSERT_TRUE(x)` | x is truthy |
| `ASSERT_FALSE(x)` | x is falsy |
| `ASSERT_NULL(x)` | x == NULL |
| `ASSERT_NOT_NULL(x)` | x != NULL |
| `ASSERT_FLOAT_EQ(a, b, eps)` | |a-b| < eps |

---

## CI Integration

```bash
# In CI script
make test && echo "✓ Tests passed" || exit 1
make bench | tee benchmark_results.txt
```

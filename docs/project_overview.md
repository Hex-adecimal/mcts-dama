# MCTS Dama: Project Overview

## Panoramica

**MCTS Dama** è un'implementazione in C di un motore di gioco per la **Dama Italiana** basato sull'approccio **AlphaZero**. Il sistema combina:

- **Monte Carlo Tree Search (MCTS)** per l'esplorazione dell'albero di gioco
- **Rete Neurale Convoluzionale (CNN)** per valutazione posizionale e guida della ricerca
- **Self-Play** per generazione automatica di dati di training

Il progetto è sviluppato interamente in C11 per massimizzare le prestazioni su architettura Apple Silicon (M2), sfruttando il framework Accelerate per operazioni BLAS/SIMD.

---

## Architettura

```
┌─────────────────────────────────────────────────────────────────┐
│                         apps/                                   │
│   ┌─────────┐   ┌───────────┐   ┌──────────────┐               │
│   │   CLI   │   │ Tournament│   │ CLOP Tuning  │               │
│   └────┬────┘   └─────┬─────┘   └──────┬───────┘               │
└────────┼──────────────┼────────────────┼────────────────────────┘
         └──────────────┴───────┬────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────┐
│                         src/                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │
│  │   engine/    │◄──│   search/    │──►│   neural/    │         │
│  │ Bitboard +   │   │ MCTS + TT +  │   │ CNN 1.4M     │         │
│  │ Italian Rules│   │ Parallelism  │   │ params       │         │
│  └──────────────┘   └──────────────┘   └──────┬───────┘         │
│                            ▲                   │                 │
│                            │                   ▼                 │
│                     ┌──────┴───────────────────────┐            │
│                     │         training/            │            │
│                     │   Self-play + SGD Pipeline   │            │
│                     └──────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Statistiche Progetto

| Metrica | Valore |
|---------|--------|
| **Linee di codice C** | ~8,000 |
| **Moduli** | 5 (engine, search, neural, training, common) |
| **Unit Test** | 75 |
| **Benchmark automatici** | 20+ |
| **Parametri CNN** | ~1.43M |

---

## Benchmark Prestazionali (Apple M2)

| Operazione | Throughput | Note |
|------------|------------|------|
| Move generation | **10.7M ops/sec** | Bitboard + Lookup Tables |
| Apply move | **59.8M ops/sec** | Zobrist hash incrementale |
| CNN forward (single) | **1,568 ops/sec** | ~0.6ms latenza |
| CNN forward (batch 16) | **1,744 samples/sec** | BLAS sgemm |
| MCTS 100 nodi (Vanilla) | **1,040 ops/sec** | ~1ms per mossa |
| MCTS 1000 nodi (AlphaZero) | **~1 ops/sec** | Bottleneck: CNN |
| Training step (batch 32) | **19 batches/sec** | ~608 samples/sec |

---

## Componenti Principali

### 1. Engine Module

Implementa le regole della Dama Italiana con rappresentazione **bitboard** a 64 bit e **lookup tables** pre-calcolate per generazione mosse O(1).

→ [Documentazione dettagliata](engine_reference.md)

### 2. Search Module  

**MCTS asincrono** con parallelismo multi-thread, **virtual loss** per diversificazione, **batch inference** per efficienza CNN.

→ [Documentazione dettagliata](search_reference.md)

### 3. Neural Module

CNN con 4 layer convoluzionali (64 canali), BatchNorm+ReLU, **policy head** (256 output) e **value head** (tanh).

→ [Documentazione dettagliata](neural_reference.md)

### 4. Training Module

Pipeline AlphaZero completa: **self-play** con Dirichlet noise e temperature annealing, **SGD con momentum**, early stopping.

→ [Documentazione dettagliata](training_reference.md)

### 5. Optimizations

Parallelismo con OpenMP/Accelerate/Pthreads. Pipeline FP16/INT8 pianificata.

→ [Documentazione dettagliata](optimizations_reference.md)

---

## Tecnologie Utilizzate

| Categoria | Tecnologia | Uso |
|-----------|------------|-----|
| **Linguaggio** | C11 | Core codebase |
| **Build** | Make | Build system |
| **BLAS** | Apple Accelerate | Matrix ops, SIMD |
| **Threading** | POSIX Threads | MCTS workers |
| **Parallelismo** | OpenMP | Training/validation |
| **Testing** | Custom framework | 75 unit test |

---

## Scelte Architetturali Chiave

| Scelta | Motivazione |
|--------|-------------|
| **Bitboard 64-bit** | Operazioni parallele su tutti i pezzi, cache-friendly |
| **Arena Allocator** | Zero overhead allocazione nodi MCTS, reset O(1) |
| **Encoding Canonico** | Board flipped per il Nero → input consistente per CNN |
| **LR separati** | Policy e value head con learning rate diversi (1:10) |
| **Batch CNN Inference** | Riduce latenza sincronizzazione in MCTS parallelo |
| **Fused BN+ReLU** | +15% inference throughput |

---

## Limitazioni e Trade-off

| Limitazione | Impatto | Motivazione |
|-------------|---------|-------------|
| **No GPU/ANE** | Lento self-play | API Metal/CoreML richiedono Objective-C |
| **Policy 256 output** | Collisioni mosse complesse | Semplicità encoding |
| **4 conv layers** | Receptive field 9×9 | Bilanciamento velocità/qualità |
| **No data augmentation** | Meno dati training | Dama non simmetrica |
| **No replay buffer** | Rischio forgetting | Complessità implementativa |

---

## Documentazione Completa

| Documento | Contenuto |
|-----------|-----------|
| [engine_reference.md](engine_reference.md) | Bitboard, movegen, Zobrist hashing |
| [search_reference.md](search_reference.md) | MCTS, UCB/PUCT, parallelismo |
| [neural_reference.md](neural_reference.md) | Architettura CNN, training, encoding |
| [training_reference.md](training_reference.md) | Self-play, dataset, SGD pipeline |
| [optimizations_reference.md](optimizations_reference.md) | Parallelismo, FP16/INT8, SIMD |
| [experiments_report.md](experiments_report.md) | **Risultati training e tournament** |
| [testing.md](testing.md) | Unit test e benchmark |
| [architecture.md](architecture.md) | Struttura moduli e data flow |

---

## Quick Start

```bash
# Build
make all

# Test
make test    # 75 unit tests
make bench   # Performance benchmarks

# Play
./bin/dama play                    # Gioca contro AI
./bin/dama tournament -g 100       # Tournament 100 partite

# Training
./bin/dama selfplay -n 100 -s 800 -o data.bin
./bin/dama train -d data.bin -e 50 -o model.bin
```

---

## Riferimenti Accademici

- Silver, D. et al. (2017). *Mastering the game of Go without human knowledge*. Nature.
- Silver, D. et al. (2018). *A general reinforcement learning algorithm that masters chess, shogi, and Go*. Science.
- Browne, C. et al. (2012). *A Survey of Monte Carlo Tree Search Methods*. IEEE.

# Project Refactoring Plan

## Situazione Attuale

### Struttura Corrente

```
MCTS Dama/
├── apps/
│   ├── cli/main.c           # CLI per partite
│   └── gui/dama_gui.c       # GUI SDL
├── tools/
│   ├── training/
│   │   ├── bootstrap.c      # Genera dati con MCTS tradizionale
│   │   ├── init_weights.c   # Inizializza pesi CNN
│   │   ├── selfplay.c       # Self-play con CNN
│   │   └── trainer.c        # Training loop
│   ├── evaluation/
│   │   ├── compare.c        # Confronta 2 modelli
│   │   ├── tournament.c     # Torneo round-robin
│   │   └── tuner.c          # SPSA tuning
│   └── data/
│       ├── inspector.c      # Ispeziona dataset
│       └── merger.c         # Unisce dataset
├── scripts/
│   ├── parallel_selfplay.sh # Script bash self-play
│   └── train_loop.sh        # Script bash training loop
├── src/
│   ├── core/                # Logica gioco (4 file)
│   ├── mcts/                # MCTS engine (15 file)
│   └── nn/                  # Neural network (8 file)
└── test/                    # Test (11 file)
```

### Problemi Identificati

1. **Troppi eseguibili separati** (9 tools + 2 apps = 11 binari)
2. **Script bash** che orchestrano tools C (meglio un solo tool con subcomandi)
3. **Duplicazione logica** tra tools simili (compare vs tournament)
4. **Directory data/models/logs** non standardizzata
5. **Troppi file in src/mcts/** (15 file per un modulo)

---

## Piano di Refactoring

### Fase 1: Consolidare Tools in Un Unico Eseguibile

**Obiettivo:** Sostituire 9 tools separati con un singolo `dama-cli` con subcomandi.

#### Prima

```bash
./bin/trainer --epochs 10
./bin/selfplay --games 100
./bin/tournament -n 1000
./bin/inspector data/game1.bin
```

#### Dopo

```bash
./bin/dama train --epochs 10
./bin/dama selfplay --games 100
./bin/dama tournament -n 1000
./bin/dama data inspect game1.bin
```

#### Struttura Proposta

```
apps/
├── cli/
│   ├── main.c              # Entry point con parser subcomandi
│   ├── cmd_train.c         # Subcomando train (unifica trainer+selfplay)
│   ├── cmd_tournament.c    # Subcomando tournament
│   ├── cmd_data.c          # Subcomando data (inspect, merge)
│   ├── cmd_tune.c          # Subcomando tune (SPSA)
│   └── cmd_play.c          # Subcomando play (partita vs AI)
└── gui/
    └── dama_gui.c          # GUI resta separata
```

**File da eliminare/fondere:**

| File Attuale | Destino |
|--------------|---------|
| tools/training/trainer.c | → cmd_train.c |
| tools/training/selfplay.c | → cmd_train.c |
| tools/training/bootstrap.c | → cmd_train.c (--bootstrap flag) |
| tools/training/init_weights.c | → cmd_train.c (--init flag) |
| tools/evaluation/tournament.c | → cmd_tournament.c |
| tools/evaluation/compare.c | → ❌ ELIMINA (tournament fa lo stesso) |
| tools/evaluation/tuner.c | → cmd_tune.c |
| tools/data/inspector.c | → cmd_data.c |
| tools/data/merger.c | → cmd_data.c |
| apps/cli/main.c | → ❌ SOSTITUITO dal nuovo main.c |

---

### Fase 2: Eliminare Script Bash

**Obiettivo:** Integrare la logica degli script nel tool C.

#### Scripts da eliminare

```
scripts/parallel_selfplay.sh  → "dama train --parallel 4"
scripts/train_loop.sh        → "dama train --loop --eval-interval 5"
```

---

### Fase 3: Semplificare src/mcts/

**Obiettivo:** Ridurre 15 file a struttura più gestibile.

#### Dimensioni Attuali

| File | Righe (~) | Bytes |
|------|-----------|-------|
| mcts.c | ~600 | 22KB |
| expansion.c | ~250 | 10KB |
| selection.c | ~130 | 5KB |
| simulation.c | ~100 | 4KB |
| mcts_internal.c | ~90 | 3.5KB |
| mcts_presets.c | ~80 | 3KB |
| backprop.c | ~50 | 2KB |

#### Opzione A: Minimalista (3 file)

```
src/mcts/
├── mcts.c/h          # API + worker + selection + expansion + backprop (~800 righe)
├── mcts_types.h      # Tutte le strutture + presets + arena
└── simulation.c      # Solo rollout (se serve vanilla, altrimenti elimina)
```

✅ **Pro:** Ultra-semplice, tutto in un posto
❌ **Contro:** mcts.c diventa molto grande

---

#### Opzione B: Per Responsabilità (4 file)

```
src/mcts/
├── mcts.c/h          # API pubblica + worker thread
├── mcts_tree.c/h     # Selection + Expansion + Backprop (logica albero)
├── mcts_types.h      # Strutture + Arena + TT + Presets
└── mcts_rollout.c    # Simulation/Rollout (solo se usato)
```

✅ **Pro:** Chiara separazione API vs internals
❌ **Contro:** mcts_types.h diventa grande

---

#### Opzione C: Per Fase Algoritmo (5 file)

```
src/mcts/
├── mcts.c/h          # API + worker + orchestrazione
├── mcts_search.c     # Selection + Expansion (tree traversal)
├── mcts_eval.c       # Backprop + Simulation (valutazione)
├── mcts_config.c/h   # Presets + Types + Arena + TT
└── (mcts_types.h inlined in mcts_config.h)
```

✅ **Pro:** Bilanciato, ogni file ha ~200 righe
❌ **Contro:** Separa backprop da selection (che sono correlati in traversal)

---

#### Opzione D: Ibrido Pratico (4 file)

```
src/mcts/
├── mcts.c/h          # API + worker + search loop completo
├── mcts_core.c/h     # Selection + Expansion + Backprop (inline-able)
├── mcts_config.h     # Types + Presets + Constants (header-only)
└── mcts_memory.c/h   # Arena + TT (gestione memoria)
```

✅ **Pro:** Chiara separazione logica vs memoria
❌ **Contro:** Ancora 4 file

---

#### ⭐ Raccomandazione

**Opzione B** offre il miglior bilanciamento tra semplicità e manutenibilità.

---

### Fase 4: Spostare in Legacy

**Obiettivo:** Pulire il progetto principale spostando codice non essenziale.

```
legacy/
├── compare.c            # Confronto 2 modelli (ridondante)
├── test/                # Vecchi test (se non più usati)
└── docs/old/            # Documentazione obsoleta
```

---

### Fase 5: Standardizzare Directory Dati

**Obiettivo:** Struttura chiara per artifacts.

```
output/                  # Tutto l'output generato
├── data/                # Dataset training
│   ├── raw/             # .bin grezzi
│   └── merged/          # Dataset consolidati
├── models/              # Pesi CNN
│   ├── checkpoints/     # Salvataggi intermedi
│   └── final/           # Modelli finali
└── logs/                # Log di esecuzione
```

---

## Struttura Finale Proposta

```
MCTS Dama/
├── src/
│   ├── core/            # 4 file (game, movegen)
│   ├── mcts/            # 6 file (ridotto da 15)
│   └── nn/              # 8 file (CNN)
├── apps/
│   ├── cli/             # 6 file (main + 5 subcomandi)
│   └── gui/             # 1 file (dama_gui.c)
├── output/              # Generato a runtime
│   ├── data/
│   ├── models/
│   └── logs/
├── docs/                # Documentazione
├── legacy/              # Codice deprecato
├── Makefile
└── README.md
```

---

## Riepilogo Cambiamenti

| Metrica | Prima | Dopo |
|---------|-------|------|
| Eseguibili | 11 | 2 (dama, dama_gui) |
| File in tools/ | 9 | 0 |
| File in src/mcts/ | 15 | 6 |
| Script bash | 2 | 0 |
| Directory top-level | 9 | 6 |

---

## Ordine di Esecuzione

1. **Fase 1** - Consolidare tools (più impatto)
2. **Fase 5** - Standardizzare directory (preparazione)
3. **Fase 2** - Eliminare scripts (dipende da Fase 1)
4. **Fase 3** - Semplificare MCTS (indipendente)
5. **Fase 4** - Spostare legacy (cleanup finale)

---

## Stima Tempi

| Fase | Ore Stimate |
|------|-------------|
| Fase 1 | 8-12h |
| Fase 2 | 2-4h |
| Fase 3 | 4-6h |
| Fase 4 | 1-2h |
| Fase 5 | 1-2h |
| Testing | 4-6h |
| **Totale** | **20-32h** |

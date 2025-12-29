---
trigger: model_decision
description: When creating new files, reorganizing code, or deciding where to put new features
---

# Project Organization Guidelines

## 📏 File Size Balance

Mantieni un equilibrio tra lunghezza e numero di file:

| Categoria | Righe Target | Azione se supera |
|-----------|--------------|------------------|
| **Header (.h)** | < 150 righe | Separa in sotto-header per responsabilità |
| **Implementazione (.c)** | 200-400 righe | Ideale |
| **Implementazione (.c)** | 400-600 righe | Considera split se logica separabile |
| **Implementazione (.c)** | > 600 righe | **Split obbligatorio** per responsabilità |

### Quando splitare un file:
1. ✅ Split se ci sono **2+ responsabilità distinte** (es: `mcts.c` → `mcts_tree.c` + `mcts_rollout.c`)
2. ✅ Split se una feature aggiunta è **autocontenuta** (es: caching, serialization)
3. ❌ NON splittare solo per rispettare il limite righe se la logica è coesa

### Naming per split:
- Usa prefisso comune: `mcts_*.c`, `cnn_*.c`
- Suffissi descrittivi: `_tree`, `_rollout`, `_inference`, `_training`

---

## 📁 Struttura Cartelle per Layer

```
_src/
├── core/           # Logica di gioco pura (game, movegen)
├── mcts/           # MCTS: albero, selezione, rollout
├── nn/             # Neural network: modello, training, inference
├── debug/          # ⭐ NUOVO: funzioni di stampa/debug
│   ├── print_board.c
│   ├── print_tree.c
│   └── debug.h     # Header comune per tutte le stampe
└── params.h        # Parametri globali condivisi
```

### Regole per `debug/`:
- Contiene **SOLO** funzioni di visualizzazione/stampa
- Nessuna logica di gioco o algoritmi
- Può includere: `print_board()`, `dump_tree()`, `log_move()`, `visualize_cnn()`
- Compilato solo in modalità DEBUG (usa `#ifdef DEBUG`)

---

## 💾 Organizzazione Dati Persistenti

Tutti i file generati vanno in `out/` con sottocartelle semplici:

```
out/
├── models/         # Checkpoint NN (.bin, .weights)
│   └── cnn_v{N}.bin
├── data/           # Dataset di training (.dat, .csv)
│   └── selfplay_{date}.dat
├── logs/           # Log di training/tournament
│   └── tournament_{date}.log
└── results/        # Risultati finali (ELO, statistiche)
    └── elo_ratings.json
```

### Convenzioni di naming:
- **Modelli**: `{tipo}_v{versione}.bin` (es: `cnn_v3.bin`)
- **Dataset**: `{source}_{YYYYMMDD}.dat` (es: `selfplay_20241229.dat`)
- **Logs**: `{tipo}_{YYYYMMDD_HHMM}.log`
- **NON** creare sottocartelle ulteriori

---

## ✅ Checklist per nuovi file

Prima di creare un file, chiediti:

1. [ ] **Esiste già un file** dove questa logica appartiene?
2. [ ] Il file esistente **supererebbe 600 righe**? → Split
3. [ ] È codice di **debug/stampa**? → Va in `_src/debug/`
4. [ ] È un **file generato** (modello, log, dataset)? → Va in `out/{tipo}/`
5. [ ] Il nome segue il pattern `{modulo}_{responsabilità}.c`?

---

## 🚫 Anti-pattern da evitare

- ❌ File > 800 righe
- ❌ Funzioni di print sparse in file di logica
- ❌ Dati salvati nella root o in cartelle random
- ❌ Nomi generici (`utils.c`, `helpers.c`, `misc.c`)
- ❌ Header che includono altri header in catena lunga

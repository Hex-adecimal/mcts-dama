# Core Game Engine Optimizations

## Obiettivo

Ottimizzare il motore di gioco Dama per massimizzare la velocità di generazione mosse e applicazione stati.

## Situazione Attuale

```
generate_moves():     ~2-5 μs/chiamata
apply_move():         ~0.5-1 μs/chiamata
Totale MCTS iter:     ~50-100 μs
```

---

## Ottimizzazione 1: Lookup Tables per Mosse

### Descrizione

Pre-calcolate tabelle statiche per mosse semplici da ogni casella.

### Implementazione

#### 1.1 Struttura Dati (`movegen.h`)

```c
// Mosse semplici pre-calcolate
extern uint64_t pawn_moves_white[32];  // Direzioni avanti
extern uint64_t pawn_moves_black[32];
extern uint64_t king_moves[32];         // Tutte le direzioni

// Inizializzazione (chiamata una volta)
void init_move_tables(void);
```

#### 1.2 Generazione Tabelle

```c
void init_move_tables(void) {
    for (int sq = 0; sq < 32; sq++) {
        uint64_t pos = 1ULL << sq;
        
        // Pedine bianche: diagonali avanti
        pawn_moves_white[sq] = 
            ((pos >> 4) & NOT_RIGHT_EDGE) |  // Avanti-sinistra
            ((pos >> 5) & NOT_LEFT_EDGE);    // Avanti-destra
        
        // Dame: tutte e 4 le direzioni
        king_moves[sq] = 
            pawn_moves_white[sq] | 
            pawn_moves_black[sq];
    }
}
```

#### 1.3 Uso in generate_moves

```c
// PRIMA (calcolo dinamico)
if (can_move_left) targets |= (pos >> 4);
if (can_move_right) targets |= (pos >> 5);

// DOPO (lookup)
targets = pawn_moves_white[sq] & empty;
```

### Guadagno Stimato

| Operazione | Prima | Dopo | Speedup |
|------------|-------|------|---------|
| Mosse semplici | 100ns | 10ns | **10x** |
| Impatto totale | - | - | **15-25%** |

---

## Ottimizzazione 2: Magic Bitboards per Catture Dame

### Descrizione

Le catture con dame lungo diagonali richiedono scanning. Magic bitboards permettono lookup O(1).

### Implementazione

#### 2.1 Struttura Magic

```c
typedef struct {
    uint64_t mask;      // Caselle rilevanti sulla diagonale
    uint64_t magic;     // Numero magico
    int shift;          // Bits da shiftare
    uint64_t *attacks;  // Tabella attacchi
} MagicEntry;

extern MagicEntry king_magic[32];
```

#### 2.2 Lookup

```c
uint64_t king_attacks(int square, uint64_t occupied) {
    MagicEntry *m = &king_magic[square];
    uint64_t idx = ((occupied & m->mask) * m->magic) >> m->shift;
    return m->attacks[idx];
}

// Uso
uint64_t targets = king_attacks(sq, all_pieces);
uint64_t captures = targets & enemy;
```

#### 2.3 Generazione Magic Numbers

Richiede pre-calcolo offline (può richiedere minuti):

```c
// Script separato per trovare magic numbers
uint64_t find_magic(int sq, int bits) {
    while (1) {
        uint64_t magic = random_sparse_uint64();
        if (test_magic(sq, magic, bits))
            return magic;
    }
}
```

### Guadagno Stimato

| Operazione | Prima | Dopo | Speedup |
|------------|-------|------|---------|
| Catture dame | 500ns | 20ns | **25x** |
| Impatto totale | - | - | **30-40%** |

### Complessità

⚠️ **ALTA** - Richiede generazione magic numbers e debugging complesso.

---

## Ottimizzazione 3: Incremental Move Update

### Descrizione

Invece di rigenerare tutte le mosse dopo ogni apply_move, aggiorna solo le mosse affette.

### Implementazione

#### 3.1 Tracking Mosse per Casella

```c
typedef struct {
    GameState state;
    MoveList moves_from[32];  // Mosse che partono da ogni casella
    uint64_t moves_to_mask;   // Bit mask delle caselle destinazione
} IncrementalState;
```

#### 3.2 Update Incrementale

```c
void incremental_apply_move(IncrementalState *s, Move *m) {
    int from = m->path[0];
    int to = m->path[m->length > 0 ? m->length : 1];
    
    // 1. Applica mossa base
    apply_move(&s->state, m);
    
    // 2. Invalida mosse da/verso caselle affette
    invalidate_moves_from(s, from);
    invalidate_moves_from(s, to);
    
    // 3. Rigenera solo mosse per caselle adiacenti
    for (int neighbor : adjacent[from])
        regenerate_moves_from(s, neighbor);
    for (int neighbor : adjacent[to])
        regenerate_moves_from(s, neighbor);
}
```

### Guadagno Stimato

| Scenario | Prima | Dopo | Speedup |
|----------|-------|------|---------|
| Early game (molti pezzi) | 3μs | 2μs | 1.5x |
| Late game (pochi pezzi) | 2μs | 0.3μs | **7x** |
| Media pesata | - | - | **2-3x** |

### Note

- Maggior beneficio in endgame (MCTS esplora più profondamente)
- Richiede gestione attenta della consistenza

---

## Ottimizzazione 4: SIMD con ARM NEON

### Descrizione

Usa istruzioni vettoriali per processare multiple operazioni bitboard in parallelo.

### Implementazione

#### 4.1 Operazioni Parallele

```c
#include <arm_neon.h>

// Processa 2 bitboard contemporaneamente
void process_both_colors(GameState *s, uint64_t *white_moves, 
                         uint64_t *black_moves) {
    // Carica entrambi i bitboard
    uint64x2_t pieces = vld1q_u64(&s->white_pieces);
    
    // Shift parallelo
    uint64x2_t left = vshrq_n_u64(pieces, 4);
    uint64x2_t right = vshrq_n_u64(pieces, 5);
    
    // Mask e combina
    uint64x2_t moves = vorrq_u64(left, right);
    
    // Store risultati
    vst1q_u64(white_moves, moves);
}
```

#### 4.2 Population Count Parallelo

```c
int count_pieces_simd(uint64_t white, uint64_t black) {
    uint64x2_t v = {white, black};
    uint8x16_t bytes = vreinterpretq_u8_u64(v);
    uint8x16_t counts = vcntq_u8(bytes);
    return vaddvq_u8(counts);  // Somma tutti i bit
}
```

### Guadagno Stimato

| Operazione | Prima | Dopo | Speedup |
|------------|-------|------|---------|
| Bit operations | 1x | 2x | 2x |
| popcount | 2 calls | 1 call | 2x |
| Impatto totale | - | - | **10-20%** |

---

## Riepilogo e Priorità

| Ottimizzazione | Speedup | Difficoltà | Priorità |
|----------------|---------|------------|----------|
| Lookup Tables | 15-25% | ⭐ Bassa | 🥇 1 |
| SIMD NEON | 10-20% | ⭐⭐ Media | 🥈 2 |
| Incremental Update | 50-200% | ⭐⭐ Media | 🥉 3 |
| Magic Bitboards | 30-40% | ⭐⭐⭐ Alta | 4 |

---

## File da Modificare

| File | Modifica |
|------|----------|
| `src/core/movegen.h` | Dichiarazioni tabelle |
| `src/core/movegen.c` | Implementazione lookup/SIMD |
| `src/core/game.h` | Struttura IncrementalState |
| `src/core/game.c` | apply_move incrementale |
| `tools/generate_magic.c` | **NUOVO** - Generatore magic numbers |

---

## Stima Tempi

| Fase | Ore Stimate |
|------|-------------|
| Lookup Tables | 2-4h |
| SIMD Integration | 4-6h |
| Incremental Update | 6-10h |
| Magic Bitboards | 10-15h |
| Testing + Benchmark | 4-6h |
| **Totale** | **26-41h** |

---

## Metriche di Successo

1. **Baseline:** Misura MCTS iter/sec attuale
2. **Target:** ≥50% aumento iter/sec combinando ottimizzazioni
3. **Validazione:** Risultati partite identici pre/post ottimizzazione

# Core Module Documentation

## Italian Checkers (Dama Italiana) Engine

Questo documento descrive l'architettura e le scelte implementative del modulo engine per la Dama Italiana.

---

## Architettura

```
src/engine/
├── game.c        # Inizializzazione, esecuzione mosse, Zobrist hashing
├── movegen.c     # Lookup tables, generazione mosse/catture
├── endgame.c     # Generazione posizioni endgame per training
└── cli_view.c    # Output CLI formattato

include/dama/engine/
├── game.h        # Tipi, costanti, macro, stato di gioco
├── movegen.h     # Interfaccia generazione mosse
└── endgame.h     # Interfaccia endgame generator
```

---

## 1. Rappresentazione della Scacchiera

### Bitboard (64-bit)

Ogni tipo di pezzo è rappresentato da un intero a 64 bit dove ogni bit corrisponde a una casella:

```
Bit 0 = A1    Bit 7 = H1
Bit 8 = A2    ...
Bit 56 = A8   Bit 63 = H8
```

**Vantaggi:**

- Operazioni parallele su tutti i pezzi simultaneamente
- Operazioni AND/OR/XOR in tempo O(1)
- Cache-friendly (64 bit = 1 registro CPU)

**Struttura GameState:**

```c
typedef struct {
    Bitboard piece[2][2];   // [WHITE/BLACK][PAWN/LADY]
    Color current_player;
    uint8_t moves_without_captures;  // Per regola 40 mosse
    uint64_t hash;                   // Zobrist hash
} GameState;
```

---

## 2. Macro Helper per Bitboard

```c
#define BIT(sq)           (1ULL << (sq))      // Singolo bit
#define TEST_BIT(bb, sq)  ((bb) & BIT(sq))    // Test presenza
#define SET_BIT(bb, sq)   ((bb) |= BIT(sq))   // Inserisci
#define CLEAR_BIT(bb, sq) ((bb) &= ~BIT(sq))  // Rimuovi
#define POP_LSB(bb)       ((bb) &= (bb) - 1)  // Rimuovi bit più basso
```

**Operazioni coordinate:**

```c
#define ROW(sq)           ((sq) / 8)
#define COL(sq)           ((sq) % 8)
#define SQUARE(row, col)  ((row) * 8 + (col))
```

---

## 3. Direzioni Diagonali

Le mosse diagonali sono codificate come offset sull'indice:

| Direzione | Step | Jump | Significato |
|-----------|------|------|-------------|
| NE (↗)    | +9   | +18  | Riga +1, Colonna +1 |
| NW (↖)    | +7   | +14  | Riga +1, Colonna -1 |
| SE (↘)    | -7   | -14  | Riga -1, Colonna +1 |
| SW (↙)    | -9   | -18  | Riga -1, Colonna -1 |

```c
#define OFFSET_NE  (+9)
#define OFFSET_NW  (+7)
#define OFFSET_SE  (-7)
#define OFFSET_SW  (-9)
```

---

## 4. Zobrist Hashing

Tecnica per calcolare hash incrementale dello stato:

```c
uint64_t zobrist_keys[2][2][64];  // [color][piece_type][square]
uint64_t zobrist_black_move;       // XOR quando tocca al nero
```

**Inizializzazione:**

- PRNG XorShift con seed fisso per riproducibilità
- Genera chiavi casuali per ogni combinazione pezzo/casella

**Aggiornamento incrementale:**

```c
// Rimuovi pezzo da A1
hash ^= zobrist_keys[WHITE][PAWN][A1];
// Aggiungi pezzo su B2
hash ^= zobrist_keys[WHITE][PAWN][B2];
// Cambio turno
hash ^= zobrist_black_move;
```

**Vantaggi:**

- O(1) per aggiornamento (vs O(n) ricalcolo completo)
- Ideale per transposition table in MCTS

---

## 5. Lookup Tables

Tabelle pre-calcolate per evitare controlli boundary runtime:

```c
static Bitboard PAWN_MOVE_TARGETS[2][64][2];   // [color][from][dir]
static Bitboard LADY_MOVE_TARGETS[64][4];      // [from][dir]
static Bitboard JUMP_LANDING[64][4];           // Casella di atterraggio
static Bitboard JUMP_OVER_SQ[64][4];           // Casella saltata
static Bitboard CAN_JUMP_FROM[64];             // Early exit optimization
```

**Inizializzazione (`init_move_tables`):**

- Calcola tutte le mosse legali da ogni casella
- Gestisce bordi automaticamente (nessun wrapping)
- O(64 × 4) = O(256) una tantum

---

## 6. Generazione Mosse

### Mosse Semplici (`generate_simple_moves`)

```c
Bitboard pawns = state->piece[us][PAWN];
while (pawns) {
    int from = __builtin_ctzll(pawns);  // Trova primo pezzo
    // Controlla target da lookup table
    for (int dir = 0; dir < NUM_PAWN_DIRS; dir++) {
        if (PAWN_MOVE_TARGETS[us][from][dir] & empty) {
            add_simple_move(...);
        }
    }
    POP_LSB(pawns);  // Passa al prossimo pezzo
}
```

### Catture (`generate_captures`)

Usa **CaptureContext** per passare stato alla ricorsione:

```c
typedef struct {
    const GameState *state;
    MoveList *list;
    int path[13];           // Caselle visitate
    int captured[12];       // Caselle catturate
    Bitboard all_enemy;     // Cache: enemy_pieces | enemy_ladies
    Bitboard occupied;
    uint8_t is_lady;
} CaptureContext;
```

**Algoritmo DFS ricorsivo (`find_captures`):**

1. Controlla ogni direzione possibile
2. Verifica presenza nemico (usando cache `all_enemy`)
3. Verifica casella di atterraggio vuota
4. Se valido: salva path, rimuovi pezzo nemico, ricorri
5. Se promozione: termina catena
6. Se nessuna continuazione: salva mossa

---

## 7. Regole Italiane (Priorità Catture)

La Dama Italiana richiede di scegliere la cattura "migliore":

| Priorità | Regola |
|----------|--------|
| 1 | Catena più lunga |
| 2 | Con dama (se possibile) |
| 3 | Che cattura più dame |
| 4 | Prima dama catturata il prima possibile |

**Implementazione (`calculate_score` + `filter_moves`):**

```c
int score = (length << 24) | (is_lady_move << 20) | 
            (captured_ladies_count << 10) | first_captured_is_lady;
```

Poi si tiene solo il punteggio massimo.

---

## 8. Ottimizzazioni Implementate

| Tecnica | Descrizione | Guadagno |
|---------|-------------|----------|
| **Lookup Tables** | Pre-calcolo mosse | ~20-30% |
| **CAN_JUMP_FROM cache** | Early exit se nessun salto possibile | ~5-10% |
| **all_enemy cache** | Evita OR ripetuto in ricorsione | ~3-5% |
| **`__builtin_prefetch`** | Hint CPU per prossimi dati | ~2-5% |
| **uint8_t per contatori** | Meno memoria, migliore cache | ~1-2% |

---

## 9. Funzioni Pubbliche

### game.h / game.c

| Funzione | Descrizione |
|----------|-------------|
| `zobrist_init()` | Inizializza tabelle hash |
| `init_game(state)` | Prepara posizione iniziale |
| `apply_move(state, move)` | Esegue mossa, aggiorna hash |
| `print_board(state)` | Debug: stampa scacchiera ASCII |

### movegen.h / movegen.c

| Funzione | Descrizione |
|----------|-------------|
| `init_move_tables()` | Prepara lookup tables |
| `generate_moves(state, list)` | Genera tutte le mosse legali |
| `generate_simple_moves(state, list)` | Solo mosse non-cattura |
| `generate_captures(state, list)` | Solo catture (senza filtro priorità) |
| `is_square_threatened(state, sq)` | Controlla se casella è attaccata |

---

## 10. Complessità Computazionale

| Operazione | Complessità |
|------------|-------------|
| `apply_move` | O(k) dove k = lunghezza catena |
| `generate_simple_moves` | O(n) dove n = numero pezzi |
| `generate_captures` | O(n × b^d) dove b = branching, d = profondità catena |
| `filter_moves` | O(m) dove m = numero mosse cattura |

In pratica, la generazione mosse è molto veloce grazie ai bitboard e lookup tables.

---

## Riferimenti

- [Chess Programming Wiki - Bitboards](https://www.chessprogramming.org/Bitboards)
- [Zobrist Hashing](https://www.chessprogramming.org/Zobrist_Hashing)
- Regole ufficiali Dama Italiana (FID)

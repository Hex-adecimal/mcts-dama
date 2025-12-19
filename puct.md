# PUCT - Predictor Upper Confidence Trees

Guida completa all'implementazione di PUCT con rete neurale per MCTS.

---

## 1. Teoria

### 1.1 Da UCB1 a PUCT

**UCB1** (usato in MCTS classico):

$$UCB1(s,a) = Q(s,a) + C \cdot \sqrt{\frac{\ln N}{n}}$$

- Esplora nodi uniformemente (ogni mossa ha stessa probabilità iniziale)
- Non usa conoscenza del dominio

**PUCT** (usato in AlphaZero):

$$PUCT(s,a) = Q(s,a) + c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{N}}{1 + n}$$

- $P(s,a)$ = **prior probability** da rete neurale
- Esplora prima le mosse che la rete considera promettenti

### 1.2 Componenti

| Simbolo | Significato |
|---------|-------------|
| $Q(s,a)$ | Valore medio dell'azione = $\frac{score}{visits}$ |
| $P(s,a)$ | Prior dalla policy network |
| $N(s)$ | Visite del nodo padre |
| $n = N(s,a)$ | Visite del nodo figlio |
| $c_{puct}$ | Costante di esplorazione (~1.5) |

### 1.3 Rete Neurale (MLP)

```text
GameState → Encoder → [Feature Vector]
                           ↓
                        [MLP]
                           ↓
              ┌────────────┴────────────┐
              ↓                         ↓
         Policy Head               Value Head
         (64 logits)               (1 scalare)
              ↓                         ↓
          softmax                    tanh
              ↓                         ↓
         P(s,a)                    V(s) ∈ [-1,1]
```

---

## 2. File Creati

### 2.1 `src/nn.h`

Strutture dati principali:

```c
// Pesi della rete
typedef struct {
    float *w1, *b1;  // Input → Hidden
    float *w2, *b2;  // Hidden → Hidden
    float *wp, *bp;  // → Policy
    float *wv, *bv;  // → Value
    // + gradienti per training
} NNWeights;

// Output
typedef struct {
    float policy[64];  // P(s,a) per ogni mossa
    float value;       // V(s)
} NNOutput;

// Sample per training
typedef struct {
    GameState state;
    float target_policy[64];  // π da MCTS
    float target_value;       // z = risultato
} TrainingSample;
```

### 2.2 `src/nn.c`

Funzioni da implementare:

| Funzione | Cosa fa |
|----------|---------|
| `nn_init()` | Alloca memoria, init random |
| `nn_encode_state()` | GameState → float[] |
| `nn_forward()` | Calcola policy + value |
| `nn_backward()` | Backpropagation |
| `nn_train_step()` | SGD su batch |
| `nn_get_move_prior()` | Mossa → $P(s,a)$ |

### 2.3 `tools/trainer.c`

Loop di training:

1. Self-play con MCTS+PUCT
2. Raccogli $(state, \pi, z)$ samples
3. Train con SGD
4. Salva checkpoint

---

## 3. File Modificati

### 3.1 `src/mcts.h`

```c
// Aggiunti a MCTSConfig:
int use_puct;          // Flag per abilitare
double puct_c;         // Costante (~1.5)
void *nn_weights;      // Puntatore a NNWeights
```

### 3.2 `src/mcts.c`

Nuova funzione:

```c
static double calculate_puct(Node *child, MCTSConfig config, float prior) {
    double q = child->score / child->visits;
    double u = config.puct_c * prior * sqrt(parent->visits) / (1 + child->visits);
    return q + u;
}
```

Routing in `calculate_ucb1_score()`:

```c
if (config.use_puct && config.nn_weights) {
    float prior = nn_get_move_prior(...);
    return calculate_puct(child, config, prior);
} else if (config.use_ucb1_tuned) {
    // ...
}
```

### 3.3 `src/params.h`

```c
#define PUCT_C  1.5  // Costante esplorazione
```

### 3.4 `Makefile`

```makefile
# nn.c aggiunto a tutti i target
SRCS = main.c src/game.c src/mcts.c src/debug.c src/nn.c

# Nuovo target
trainer: ...
    tools/trainer.c src/nn.c ...
```

---

## 4. Come Usare PUCT

### 4.1 Configurazione

```c
// In main.c o dove configuri MCTS
NNWeights weights;
nn_init(&weights, 128, 256, 64);
nn_load_weights(&weights, "weights.bin");

MCTSConfig config = {
    .use_puct = 1,
    .puct_c = PUCT_C,
    .nn_weights = &weights,
    // altri parametri...
};
```

### 4.2 Disabilitare PUCT (fallback a UCB1)

```c
MCTSConfig config = {
    .use_puct = 0,  // Disabilitato
    // oppure
    .use_puct = 1,
    .nn_weights = NULL,  // Nessun peso → fallback
};
```

---

## 5. Training

### 5.1 Self-Play Loop

```text
Per ogni partita:
  1. Inizia GameState
  2. Per ogni mossa:
     a. MCTS search con PUCT
     b. Salva (state, visit_counts)
  3. Determina vincitore
  4. Per ogni sample:
     target_value = +1 se vincitore, -1 se perdente
```

### 5.2 Training Step

```text
Per ogni batch:
  1. Forward pass → predizioni
  2. Calcola loss: L = L_π + L_v
  3. Backward pass → gradienti
  4. Update: w -= lr * grad
```

### 5.3 Loss Functions

**Policy Loss** (Cross-Entropy):

$$L_\pi = -\sum_a \pi(a) \cdot \log P(s,a)$$

**Value Loss** (MSE):

$$L_v = (V(s) - z)^2$$

**Total Loss**:

$$L = L_\pi + L_v$$

---

## 6. Build & Run

```bash
# Compila tutto
make clean && make

# Compila trainer
make trainer

# Esegui training
./bin/trainer
```

---

## 7. Differenze UCB1 vs PUCT

| Aspetto | UCB1 | PUCT |
|---------|------|------|
| Prior | Uniforme | Da rete neurale |
| Esplorazione | $\sqrt{\frac{\ln N}{n}}$ | $P(s,a) \cdot \frac{\sqrt{N}}{1+n}$ |
| Knowledge | Nessuno | Learned |
| Training | Non serve | Necessario |
| Performance | Buona | Eccellente (con training) |

---

## 8. Riferimenti

- **AlphaGo/AlphaZero Papers**: Formula PUCT originale
- `mcts.c`: Implementazione UCB1 esistente per confronto
- `params.h`: Costanti di configurazione

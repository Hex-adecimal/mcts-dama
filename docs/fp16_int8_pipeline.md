# FP16 Training + INT8 Inference Pipeline

## Obiettivo

Ottimizzare la CNN per massimizzare la velocità di inferenza in MCTS mantenendo la qualità del modello.

## Pipeline Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  FP16 Training  │ → │   Calibration   │ → │  INT8 Inference │
│   (2x speedup)  │    │  (100 samples)  │    │   (3.5x speed)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## Fase 1: Training FP16/Mixed Precision

### Modifiche Richieste

#### 1.1 Struttura Pesi (`cnn.h`)

```c
typedef struct {
    _Float16 *conv1_weights;    // Era float*
    _Float16 *conv1_bias;
    // ... altri layer
    float *master_weights;      // Copia FP32 per accumulo gradienti
} CNNWeightsFP16;
```

#### 1.2 Forward Pass (`cnn_inference.c`)

```c
void cnn_forward_fp16(const CNNWeightsFP16 *w, 
                      const _Float16 *input,
                      CNNOutput *out) {
    // Usa istruzioni ARM NEON per FP16
    float16x8_t vec = vld1q_f16(input);
    // ...
}
```

#### 1.3 Backward Pass (`cnn_training.c`)

```c
// Accumula gradienti in FP32 (evita underflow)
for (int i = 0; i < size; i++) {
    master_weights[i] += (float)gradient_fp16[i] * learning_rate;
}
```

#### 1.4 Loss Scaling

```c
#define LOSS_SCALE 1024.0f  // Previene underflow

float scaled_loss = loss * LOSS_SCALE;
backward(scaled_loss);
// Dopo backward:
for (int i = 0; i < size; i++)
    gradients[i] /= LOSS_SCALE;
```

### Benefici Fase 1

| Metrica | Prima | Dopo |
|---------|-------|------|
| Training Speed | 1x | 1.8-2x |
| Memory Bandwidth | 100% | 50% |
| Precisione | 100% | ~99.5% |

---

## Fase 2: Post-Training Quantization (INT8)

### 2.1 Calibrazione

```c
// Raccolta statistiche per scale factors
typedef struct {
    float min_activation;
    float max_activation;
} LayerStats;

void calibrate(CNNWeightsFP16 *model, GameState *samples, int n) {
    LayerStats stats[NUM_LAYERS] = {0};
    
    for (int i = 0; i < n; i++) {
        forward_with_stats(model, &samples[i], stats);
    }
    
    // Calcola scale factors
    for (int l = 0; l < NUM_LAYERS; l++) {
        model->scales[l] = 127.0f / fmax(fabs(stats[l].min), 
                                          fabs(stats[l].max));
    }
}
```

### 2.2 Quantizzazione Pesi

```c
typedef struct {
    int8_t *conv1_weights;
    int8_t *conv1_bias;
    float scale_conv1;
    // ...
} CNNWeightsINT8;

void quantize_weights(CNNWeightsFP16 *src, CNNWeightsINT8 *dst) {
    for (int i = 0; i < size; i++) {
        float w = (float)src->conv1_weights[i];
        dst->conv1_weights[i] = (int8_t)round(w * dst->scale_conv1);
    }
}
```

### 2.3 Inference INT8

```c
void cnn_forward_int8(const CNNWeightsINT8 *w,
                      const int8_t *input,
                      CNNOutput *out) {
    int32_t acc = 0;
    
    // ARM NEON per INT8
    int8x16_t va = vld1q_s8(input);
    int8x16_t vw = vld1q_s8(w->conv1_weights);
    
    // Dot product
    acc = vaddvq_s32(vmull_s8(va, vw));
    
    // De-quantizza
    out->value = (float)acc * w->scale_conv1 * input_scale;
}
```

### Benefici Fase 2

| Metrica | FP16 | INT8 |
|---------|------|------|
| Inference Speed | 2x | 3.5x |
| Memoria Pesi | 50% | 25% |
| Cache Efficiency | Good | Excellent |

---

## Fase 3: Integrazione MCTS

### 3.1 Modifica Config

```c
typedef struct {
    CNNWeightsINT8 *cnn_weights_int8;  // Nuovo
    int use_int8_inference;             // Flag
} MCTSConfig;
```

### 3.2 Forward Condizionale

```c
void mcts_evaluate(Node *node, MCTSConfig *cfg, CNNOutput *out) {
    if (cfg->use_int8_inference) {
        int8_t input[CNN_INPUT_SIZE];
        quantize_input(&node->state, input, cfg->input_scale);
        cnn_forward_int8(cfg->cnn_weights_int8, input, out);
    } else {
        cnn_forward_fp16(cfg->cnn_weights, input, out);
    }
}
```

---

## File da Modificare

| File | Modifica |
|------|----------|
| `src/nn/cnn.h` | Nuove struct FP16/INT8 |
| `src/nn/cnn_core.c` | Allocazione memoria FP16 |
| `src/nn/cnn_inference.c` | Forward FP16 e INT8 |
| `src/nn/cnn_training.c` | Mixed precision + loss scaling |
| `src/mcts/mcts_types.h` | Aggiunta config INT8 |
| `tools/training/trainer.c` | Usa training FP16 |
| `tools/quantize.c` | **NUOVO** - Tool di quantizzazione |

---

## Stima Tempi

| Fase | Ore Stimate |
|------|-------------|
| FP16 Training | 8-12h |
| Calibrazione + Quantizzazione | 4-6h |
| Integrazione MCTS | 2-4h |
| Testing + Debug | 4-8h |
| **Totale** | **18-30h** |

---

## Rischi e Mitigazioni

| Rischio | Probabilità | Mitigazione |
|---------|-------------|-------------|
| NaN durante training FP16 | Media | Loss scaling, gradient clipping |
| Perdita precisione INT8 | Bassa | Calibrazione accurata |
| Instabilità numerica | Bassa | Fallback a FP32 per layer critici |

---

## Metriche di Successo

1. **Training:** <5% slowdown rispetto a FP32 in convergenza
2. **Inference:** ≥3x speedup in MCTS iter/sec
3. **Qualità:** <50 ELO drop rispetto a FP32

---
trigger: model_decision
description: When analyzing training results, discussing model performance, or evaluating new experiments
---

# Critical Training Analysis

## 🎯 Obiettivo

Quando si discutono risultati di training, **mettere sempre in dubbio** le metriche prima di trarre conclusioni. Una loss che scende NON significa automaticamente che il modello ha imparato.

---

## 🔍 Checklist di Analisi Critica

### 1. Dataset Quality
Prima di guardare la loss, analizza i **dati di training**:

| Domanda | Cosa cercare | Red Flag 🚩 |
|---------|--------------|-------------|
| Bilanciamento classi? | % vittorie W / % vittorie B / % pareggi | > 70% di una classe |
| Distribuzione mosse? | Varietà di aperture e posizioni | Posizioni ripetitive |
| Qualità partite? | Come sono state generate (random vs MCTS) | Partite troppo corte/lunghe |
| Dimensione dataset? | Numero di samples | < 10k samples per training serio |

### 2. Loss Breakdown
**MAI** guardare solo la loss totale. Analizza separatamente:

```
Total Loss = α × Policy_Loss + β × Value_Loss

Chiedi sempre:
- Policy loss sta scendendo? → Il modello impara a scegliere mosse?
- Value loss sta scendendo? → Il modello impara a valutare posizioni?
- Una scende e l'altra sale? → 🚩 Problema di bilanciamento α/β
```

### 3. Validazione Reale (NON solo metriche)
La vera domanda: **il modello gioca meglio?**

| Test | Come farlo | Cosa cercare |
|------|------------|--------------|
| **Tournament vs baseline** | 100+ partite vs MCTS vanilla | Win rate > 55% è significativo |
| **Tournament vs versione precedente** | 100+ partite vs modello N-1 | Miglioramento costante |
| **Ispezione manuale** | Guarda 5-10 partite | Mosse sensate? Blunder ovvi? |
| **Posizioni test** | Posizioni note con mossa "giusta" | Il modello le trova? |

---

## ⚠️ Trappole Comuni

### 🚩 "La loss è scesa tantissimo!"
**Domande da fare:**
- Su che dati? Training set? Validation set?
- C'è overfitting? (loss training ↓ ma validation ↑)
- Il dataset era troppo facile/ripetitivo?

### 🚩 "Policy accuracy è al 90%!"
**Domande da fare:**
- Quante mosse legali in media? (se 2-3, 90% non è impressionante)
- Accuracy su posizioni complesse vs semplici?
- Il modello sta memorizzando o generalizzando?

### 🚩 "Value prediction è precisa!"
**Domande da fare:**
- Distribution shift? (training su posizioni diverse da quelle di gioco)
- Il modello predice sempre ~0? (safe bet per dati bilanciati)
- Calibrazione: predizione 0.7 → vince 70% delle volte?

### 🚩 "Il modello è migliorato nel tournament!"
**Domande da fare:**
- Quante partite? (< 50 → varianza troppo alta)
- Contro chi? (battere random non significa nulla)
- Tempo per mossa uguale? (più tempo = più iterazioni MCTS = vantaggio sleale)

---

## 📊 Template per Discussione Risultati

Quando presenti risultati di training, includi SEMPRE:

```markdown
## Training Run: [nome/data]

### Dataset
- Source: [selfplay/tournament/mixed]
- Samples: [N]
- Distribuzione: [W%/B%/D%]
- Posizioni uniche: [N o stima]

### Metriche Training
- Policy Loss: [inizio] → [fine]
- Value Loss: [inizio] → [fine]  
- Epochs: [N]
- Learning Rate: [valore]

### Validazione
- Tournament vs [baseline]: [W-L-D] ([win rate]%)
- Partite analizzate manualmente: [osservazioni]

### ⚠️ Dubbi / Limitazioni
- [Lista di possibili problemi o bias]

### Conclusione
- [Solo dopo aver risposto ai dubbi]
```

---

## 🧠 Mindset

> "Un modello è colpevole di non aver imparato finché non si dimostra innocente."

Non celebrare mai una loss che scende. Celebra solo win rate che sale in tournament reali.

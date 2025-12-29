---
trigger: model_decision
description: When analyzing tournament results or comparing MCTS configurations
---

# Critical Tournament Analysis

## �� Obiettivo

Quando si analizzano i risultati di un tournament, **non fermarsi al win rate**. Ogni metrica racconta una storia diversa e può rivelare problemi nascosti.

---

## 📊 Metriche del Tournament e Come Interpretarle

### Leaderboard Metrics

| Metrica | Cosa Significa | Red Flag 🚩 |
|---------|----------------|-------------|
| **Points** | Vittorie + 0.5 × Pareggi | Molti pareggi → strategie difensive? |
| **Win Rate** | % vittorie su totale | < 50% vs baseline è problematico |
| **ELO** | Forza relativa stimata | Differenza < 50 ELO → non significativa |
| **Wins/Loss** | Confronto diretto | Guardare distribuzione, non solo totale |

### Per-Match Stats

| Metrica | Cosa Significa | Range Tipico | Cosa Indica |
|---------|----------------|--------------|-------------|
| **iter/move** | Iterazioni MCTS per mossa | ~node_limit | Efficienza della ricerca |
| **nodes** | Nodi albero per mossa | Varia | CNN crea più nodi (espande tutto) |
| **Depth** | Profondità media albero | 5-15 | Ricerca profonda vs superficiale |
| **Exp/move** | Espansioni per mossa | 1-100 | CNN >> Vanilla (expand all children) |
| **ch/exp** | Figli per espansione | 1 o ~15-20 | Vanilla=1, CNN=tutti i figli |

---

## 🔍 Checklist di Analisi Critica

### 1. Significatività Statistica
**Prima di tutto**: i risultati sono statisticamente significativi?

| Numero Partite | Affidabilità | Azione |
|----------------|--------------|--------|
| < 20 | ❌ Non affidabile | Aumenta partite |
| 20-50 | ⚠️ Indica trend | Conferma con più dati |
| 50-100 | ✅ Ragionevole | Puoi trarre conclusioni |
| > 100 | ✅✅ Molto affidabile | Risultati solidi |

**Formula rapida errore standard:**
```
SE ≈ 0.5 / √N
Per N=100: SE ≈ 5% → win rate 55% significa range [50%, 60%]
```

### 2. Confronti Head-to-Head
Non guardare solo il ranking finale. Analizza i singoli matchup:

- Chi batte chi? (transitività: A > B > C ma C > A?)
- Ci sono matchup "strani"? (CNN perde vs Vanilla ma batte Grandmaster?)
- Come performa contro baseline (PureVanilla)?

### 3. Efficienza vs Forza
Un modello può vincere ma essere **inefficiente**:

| Osservazione | Possibile Problema |
|--------------|-------------------|
| iter/move molto alto ma win rate medio | Spreco di compute |
| nodes/move esplosivo (CNN) | Memory pressure, potenziale slowdown |
| Depth bassa con molte iterazioni | Albero troppo largo, non profondo |

### 4. Analisi CNN vs Vanilla
Le CNN hanno pattern diversi:

```
Vanilla: iter/move ≈ nodes/move, Exp=iter, ch/exp ≈ 1
CNN:     iter/move << nodes/move, Exp << iter, ch/exp ≈ 15-20

Se CNN ha ch/exp ≈ 1 → 🚩 Non sta usando policy correctamente
Se CNN ha nodes esplosivi → 🚩 Potenziale memory issue
```

---

## ⚠️ Trappole Comuni

### 🚩 "Il modello ha ELO più alto!"
**Domande da fare:**
- Di quanto? (<50 ELO non è significativo)
- Quante partite? (piccolo N = varianza alta)
- Baseline era ragionevole? (battere random non conta)

### 🚩 "Il modello vince il 60% delle partite!"
**Domande da fare:**
- Contro chi? (sbilanciamento roster?)
- Vince sempre contro gli stessi avversari?
- Perde sempre contro un tipo specifico di avversario?

### 🚩 "Depth è molto alta, quindi è meglio!"
**Attenzione:**
- Depth alta può significare ricerca stretta, non necessariamente migliore
- Confronta depth/iter ratio tra modelli
- Un modello con depth 10 e 1000 iter può essere peggiore di depth 5 e 100 iter

### 🚩 "CNN usa più nodi quindi esplora meglio!"
**Attenzione:**
- CNN espande TUTTI i figli → naturalmente più nodi
- Nodi ≠ qualità della ricerca
- Confronta iter/move, non nodes/move tra Vanilla e CNN

---

## 📋 Template per Discussione Risultati Tournament

```markdown
## Tournament Results: [data/nome]

### Setup
- Games per pairing: [N]
- Node limit: [N]
- Time limit: [T]s
- Partecipanti: [lista]

### Leaderboard Summary
[copia tabella finale]

### Analisi Significatività
- Totale partite per player: [N]
- Margine errore stimato: ±[X]%
- Differenze ELO significative (>50): [lista coppie]

### Head-to-Head Notabili
- [Player A] vs [Player B]: [W-L-D] - [osservazione]
- ...

### Efficienza
| Player | iter/move | nodes/move | Ratio | Note |
|--------|-----------|------------|-------|------|
| ...    | ...       | ...        | ...   | ...  |

### ⚠️ Red Flags / Anomalie
- [Lista di osservazioni sospette]

### Conclusioni
- [Solo dopo aver analizzato tutto]
- [Prossimi esperimenti suggeriti]
```

---

## 🧠 Mindset

> "Un ELO alto è innocente finché non si dimostra statisticamente significativo."

Non fidarti mai di un singolo tournament. Replica i risultati prima di concludere.

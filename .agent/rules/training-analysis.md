---
trigger: model_decision
description: When analyzing training results, discussing model performance, or evaluating new experiments
---

# Training Analysis

Loss going down ≠ model learned. Question everything.

## Critical Checklist

### 1. Dataset Quality

| Question | Red Flag |
|----------|----------|
| Class balance? | >70% one class |
| Move diversity? | Repetitive positions |
| Game quality? | Too short (<20) or too long (>200) |
| Dataset size? | <10k samples for serious training |

### 2. Loss Breakdown

**Always analyze separately:**
```
Total = α×Policy_Loss + β×Value_Loss

Ask:
- Policy loss decreasing? → Learning moves?
- Value loss decreasing? → Learning positions?
- One ↓ other ↑? → 🚩 Balance issue α/β
```

### 3. Real Validation (Not Just Metrics)

| Test | How | What to Check |
|------|-----|---------------|
| **vs baseline** | 100+ games vs vanilla | Win rate >55% significant |
| **vs previous** | 100+ games vs model N-1 | Consistent improvement |
| **Manual inspection** | Watch 5-10 games | Sensible moves? Blunders? |
| **Test positions** | Known positions | Finds right moves? |

## Common Traps

- **"Loss dropped a lot!"** → On what data? Overfitting? Easy dataset?
- **"90% policy accuracy!"** → How many legal moves? (2-3 moves → 90% not impressive)
- **"Value precise!"** → Always predicts ~0? (Safe bet for balanced data)
- **"Improved in tournament!"** → How many games? (<50 = high variance)

## Report Template

```markdown
## Training: [name/date]

**Dataset**:
- Source: [selfplay/tournament/mixed]
- Samples: N
- Distribution: W%/B%/D%

**Metrics**:
- Policy Loss: [start] → [end]
- Value Loss: [start] → [end]
- Epochs: N, LR: X

**Validation**:
- Tournament vs [baseline]: W-L-D (X%)
- Manual observations: [notes]

**Doubts/Limitations**: [list concerns]

**Conclusion**: [only after addressing doubts]
```

> "A model is guilty of not learning until proven innocent."

Never celebrate loss decrease. Celebrate win rate increase.

---
trigger: model_decision
description: When analyzing tournament results or comparing MCTS configurations
---

# Tournament Analysis

Don't just look at win rate. Analyze critically.

## Key Metrics

| Metric | Meaning | Red Flag |
|--------|---------|----------|
| **Win Rate** | % wins | <50% vs baseline is bad |
| **ELO** | Relative strength | Δ<50 not significant |
| **iter/move** | MCTS iterations | Should ≈ node_limit |
| **nodes/move** | Tree nodes | CNN > Vanilla (expands all) |
| **ch/exp** | Children/expansion | Vanilla≈1, CNN≈15-20 |

## Critical Checklist

### 1. Statistical Significance
| Games | Reliability | Action |
|-------|-------------|--------|
| <20 | ❌ Not reliable | Need more |
| 20-50 | ⚠️ Trend only | Confirm |
| 50-100 | ✅ Reasonable | Can conclude |
| >100 | ✅✅ Very reliable | Solid |

### 2. Head-to-Head
- Who beats who? (Check transitivity: A>B>C but C>A?)
- Strange matchups? (CNN loses to Vanilla but beats Grandmaster?)
- How vs baseline?

### 3. Efficiency vs Strength
| Observation | Problem |
|-------------|---------|
| High iter/move, medium win rate | Wasted compute |
| Explosive nodes (CNN) | Memory pressure |
| Low depth, high iterations | Tree too wide |

### 4. CNN vs Vanilla Patterns
```
Vanilla: iter≈nodes, Exp≈iter, ch/exp≈1
CNN:     iter<<nodes, Exp<<iter, ch/exp≈15-20

🚩 If CNN has ch/exp≈1 → Not using policy correctly
🚩 If CNN nodes explosive → Memory issue
```

## Common Traps

- **"Higher ELO!"** → By how much? <50 ELO not meaningful
- **"60% win rate!"** → Against who? Balanced opponents?
- **"High depth!"** → Depth alone doesn't mean better search
- **"More nodes!"** → CNN naturally creates more nodes

## Report Template

```markdown
## Tournament: [name/date]

**Setup**: N games/pairing, M nodes, T seconds
**Players**: [list]

**Statistical Significance**: 
- Games/player: N
- Error margin: ±X%
- Significant ELO diff (>50): [pairs]

**Head-to-Head Notable**:
- PlayerA vs PlayerB: W-L-D [observation]

**Red Flags**: [list anomalies]

**Conclusion**: [after analysis only]
```

> "An ELO is innocent until proven statistically significant."

Never trust a single tournament. Replicate results.

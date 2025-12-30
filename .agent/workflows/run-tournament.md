---
description: Run a statistically valid tournament
---

# How to Run a Valid Tournament

// turbo

1. Build latest version
```bash
make clean && make
```

2. Run tournament with minimum 100 games per pairing
```bash
./build/damiera tournament \
  --games 100 \
  --nodes 800 \
  --configs "Vanilla,Grandmaster,CNN_v3" \
  > out/logs/tournament_$(date +%Y%m%d_%H%M).log
```

3. Analyze results with critical eye
- Check statistical significance (N > 50)
- Compare head-to-head, not just ELO
- Look for anomalies (transitivity violations)

4. Document in `docs/tournaments/` folder

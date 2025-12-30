---
description: Complete training iteration (selfplay → train → evaluate)
---

# Training Iteration Workflow

1. Generate self-play data
```bash
./build/damiera selfplay \
  --games 500 \
  --nodes 800 \
  --temp 1.0 \
  --dirichlet 0.25 \
  --output out/data/selfplay_$(date +%Y%m%d).dat
```

2. Train model on new data
```bash
./build/damiera train \
  --data out/data/selfplay_$(date +%Y%m%d).dat \
  --model out/models/cnn_v3.bin \
  --epochs 10 \
  --lr 0.001 \
  --output out/models/cnn_v4.bin
```

3. Evaluate vs previous version
```bash
./build/damiera tournament \
  --games 100 \
  --configs "CNN_v3,CNN_v4" \
  > out/logs/eval_v3_v4.log
```

4. If win rate >55%, promote v4 to production

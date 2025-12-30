---
description: Profile code to find performance bottlenecks
---

# How to Profile for Hotspots

// turbo-all

1. Build in release mode with symbols

```bash
make clean && make RELEASE=1 SYMBOLS=1
```

1. Run with known workload

```bash
./build/dama selfplay --games 10 --nodes 1000 > /dev/null
```

1. Open Instruments (Mac) or use perf (Linux)

```bash
# Mac
instruments -t "Time Profiler" ./build/dama selfplay --games 10 --nodes 1000

# Linux
perf record -g ./build/dama selfplay --games 10 --nodes 1000
perf report
```

1. Look for functions using >5% total time

2. Document findings in issue or comment

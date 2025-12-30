---
trigger: model_decision
description: When committing code or preparing to push changes
---

# Git Commit Hygiene

## Before Every Commit
1. **Build check**: `make clean && make` must pass
2. **No debug code**: Remove printf/debug logs, fixed seeds
3. **No commented code**: Remove unnecessary commented blocks

## Commit Message Format
```
[area] Brief description (max 50 chars)

- What changed
- Why (if not obvious)
- Breaking changes (if any)
```

**Examples**:
- `[mcts] Add transposition table to reduce duplicates`
- `[nn] Fix value head output range to [-1,1]`
- `[training] Increase batch size from 64 to 128`

## Atomic Commits
- One commit = one logical feature/fix
- Don't mix refactoring with new features
- Don't commit work-in-progress

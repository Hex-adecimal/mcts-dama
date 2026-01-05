---
trigger: model_decision
description: When writing or refactoring C code
---

# C Clean Code & Modular Programming

All identifiers and comments must be in **English**.

## 1. Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Functions, variables | `snake_case` | `user_count`, `init_game` |
| Type definitions | `PascalCase` | `GameState`, `MCTSConfig` |
| Macros, constants | `UPPER_CASE` | `MAX_MOVES`, `CNN_POLICY_SIZE` |
| Booleans | `is_`, `has_`, `can_` prefix | `is_terminal`, `has_captures` |

**Avoid:** `tmp`, `data`, `val`. Use `retry_count`, `player_id`.

---

## 2. Function Design

- **Max ~30 lines** — if longer, extract sub-logic
- **Single Responsibility** — if name contains "and", split it
- **Max 3 arguments** — use structs to group related params
- **No hidden side effects** — don't modify globals unexpectedly

---

## 3. SOLID in C

| Principle | Application |
|-----------|-------------|
| **S** (SRP) | One `.c/.h` pair per logical component |
| **O** (Open/Closed) | Function pointers in structs for extension |
| **I** (Interface Seg.) | Specific headers, no massive `globals.h` |
| **D** (Dependency Inv.) | Use opaque pointers, forward declarations |

---

## 4. Memory Safety

```c
// ✅ Always check malloc, initialize pointers
float *buf = malloc(size);
if (!buf) return ERR_MEMORY;

// ✅ Clear ownership: document who frees
// ✅ Use strncpy, snprintf — never strcpy, sprintf
```

---

## 5. Comments & Documentation

- Explain **WHY**, not WHAT — code should be self-explanatory
- Use Doxygen in headers:

```c
/**
 * @brief Checks if user has admin privileges.
 * @param user Pointer to user struct.
 * @return true if admin, false otherwise.
 */
bool is_user_admin(const User *user);
```

---

## 6. Post-Edit Checklist

- [ ] No trivial comments on obvious code
- [ ] Constants in `params.h` (global) or local scope
- [ ] Comments in English, using `// ---` headers
- [ ] No orphaned code or unused `#define`s
- [ ] Run `make test` after edits

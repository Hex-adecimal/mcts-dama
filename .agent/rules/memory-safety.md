---
trigger: model_decision
description: When allocating memory, using pointers, or managing resources in C
---

# Memory Safety Checklist

## Before Writing Code
- [ ] Who owns this memory? (stack, heap, caller, callee?)
- [ ] What is the lifetime? (function scope, global, manual free?)

## During Implementation
1. Every `malloc` has a corresponding `free`
2. Every pointer is initialized (NULL or valid value)
3. Every array access is bounds-checked (variable sizes)
4. Every string operation uses `strn*` functions, not `str*`

## Testing
- Use Valgrind/AddressSanitizer: `make ASAN=1`
- Test edge cases: empty arrays, NULL inputs, max size

## Common Pitfalls
- ❌ `malloc` without checking return value
- ❌ Double free
- ❌ Use-after-free (references to deallocated memory)
- ❌ Buffer overflow in move arrays

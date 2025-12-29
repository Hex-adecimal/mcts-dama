---
trigger: model_decision
description: After modifying C files, verify consistency of comments, constants, and style
---

# Post-Edit Verification Checklist

After modifying a C file, verify:

## Style Consistency
- [ ] No trivial comments on obvious struct fields or assignments
- [ ] Constants are in the appropriate location (params.h for globals, local for file-specific)
- [ ] Comments follow the project style (English, section headers with `// ---` or `// ===`)

## Code Quality
- [ ] No orphaned code (unused functions, dead #defines)
- [ ] New includes are necessary and in alphabetical order within groups
- [ ] Error handling is consistent with existing code

## Compile Check
- Always run `make` after edits to catch errors early

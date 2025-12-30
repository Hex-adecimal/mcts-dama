---
trigger: model_decision
description: After modifying C files, verify consistency of comments, constants, and style
---

# Post-Edit Verification

After modifying C files:

## Style
- [ ] No trivial comments on obvious fields/assignments
- [ ] Constants in right place (params.h for globals, local for file-specific)
- [ ] Comments follow style (English, `// ---` or `// ===` headers)

## Quality
- [ ] No orphaned code (unused functions, dead #defines)
- [ ] New includes necessary and alphabetically ordered
- [ ] Error handling consistent with existing code

## Compile
- [ ] Always run `make` after edits

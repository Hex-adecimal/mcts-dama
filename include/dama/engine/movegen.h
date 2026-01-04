/**
 * movegen.h - Move Generation for Italian Checkers
 * Call movegen_init() once at startup, then use movegen_generate().
 */

#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "dama/engine/game.h"

// --- Public API ---

// Initialize lookup tables. Must call once before move generation.
void movegen_init(void);

// Main entry point: generates all legal moves with Italian priority filtering
void movegen_generate(const GameState *s, MoveList *list);

// Generate simple (non-capture) moves only
void movegen_generate_simple(const GameState *s, MoveList *list);

// Generate all capture moves (no Italian priority filtering)
void movegen_generate_captures(const GameState *s, MoveList *list);

// Check if a square could be captured by opponent on their next move
int movegen_is_square_threatened(const GameState *state, int square);

#endif /* MOVEGEN_H */

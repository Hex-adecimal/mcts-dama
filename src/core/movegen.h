/**
 * movegen.h - Move Generation for Italian Checkers
 * Call init_move_tables() once at startup, then use generate_moves().
 */

#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "game.h"

// Initialize lookup tables. Must call once before move generation.
void init_move_tables(void);

// Main entry point: generates all legal moves with Italian priority filtering
void generate_moves(const GameState *s, MoveList *list);

// Generate simple (non-capture) moves only
void generate_simple_moves(const GameState *s, MoveList *list);

// Generate all capture moves (no Italian priority filtering)
void generate_captures(const GameState *s, MoveList *list);

// Check if a square could be captured by opponent on their next move
int is_square_threatened(const GameState *state, int square);

#endif /* MOVEGEN_H */

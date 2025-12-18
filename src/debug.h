#ifndef DEBUG_H
#define DEBUG_H

#include "game.h"
#include "mcts.h"

/*
 * Calculations average UCB value of all children of the root.
 * Useful for debugging.
 */
double mcts_get_avg_root_ucb(Node *root, MCTSConfig config);
void print_mcts_stats_sorted(Node *root);

/**
 * Prints the current board state to the console.
 * @param state Pointer to the GameState to display.
 */
void print_board(const GameState *state);

/**
 * Prints the algebraic coordinates (e.g., "A1") of a square index.
 * @param square_idx The 0-63 index of the square.
 */
void print_coords(int square_idx);

/**
 * Prints the list of generated moves to the console.
 * @param list Pointer to the MoveList containing moves to print.
 */
void print_move_list(MoveList *list);

/**
 * Prints a human-readable description of a move.
 */
void print_move_description(Move m);

#endif // DEBUG_H


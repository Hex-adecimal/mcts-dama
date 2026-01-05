/**
 * @file game_view.h
 * @brief Debug visualization and printing functions for GameState and Moves.
 */

#ifndef GAME_VIEW_H
#define GAME_VIEW_H

#include "dama/engine/game.h"

/**
 * @brief Print the board to stdout (ASCII representation).
 *
 * @param state Pointer to the GameState to print.
 */
void print_board(const GameState *state);

/**
 * @brief Print the coordinates of a square index.
 *
 * @param square_idx The square index (0-63).
 */
void print_coords(int square_idx);

/**
 * @brief Print the list of available moves.
 *
 * @param list Pointer to the MoveList to print.
 */
void print_move_list(const MoveList *list);

/**
 * @brief Print a description of a single move.
 *
 * @param m The Move to describe.
 */
void print_move_description(Move m);

#endif /* GAME_VIEW_H */

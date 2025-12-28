#ifndef SIMULATION_H
#define SIMULATION_H

#include "mcts_types.h"

// =============================================================================
// SIMULATION API
// =============================================================================

/**
 * Pick a "smart" move during rollout instead of purely random.
 * Uses heuristics and optional 1-ply lookahead.
 */
Move pick_smart_move(const MoveList *list, const GameState *state, int use_lookahead, MCTSConfig config);

/**
 * Simulate a random game from the given node to determine a winner.
 * @return Score from the perspective of the node's player (WIN_SCORE, LOSS_SCORE, or DRAW_SCORE).
 */
double simulate_rollout(Node *node, MCTSConfig config);

#endif // SIMULATION_H

#ifndef BACKPROP_H
#define BACKPROP_H

#include "mcts_types.h"

// =============================================================================
// BACKPROPAGATION API
// =============================================================================

/**
 * Update solver status based on children's solved states.
 */
void update_solver_status(Node *node);

/**
 * Propagate the simulation result back up the tree.
 * Updates visits and scores for all nodes in the path.
 */
void backpropagate(Node *node, double result, int use_solver);

#endif // BACKPROP_H

/**
 * mcts.h - MCTS Public API
 * 
 * This is the main header that clients should include.
 */

#ifndef MCTS_H
#define MCTS_H

#include "dama/search/mcts_types.h"  // Types, Arena, TT, Presets (all consolidated)
#include "dama/search/mcts_tree.h"   // Tree operations (selection, expansion, backprop)

// =============================================================================
// MCTS SEARCH API
// =============================================================================

/**
 * Creates the root node for the MCTS search.
 */
/**
 * Creates the root node for the MCTS search.
 */
Node* mcts_create_root(GameState state, Arena *arena, MCTSConfig config);

/**
 * Creates the root node with explicit history parent (for CNN context).
 */
Node* mcts_create_root_with_history(GameState state, Arena *arena, MCTSConfig config, Node *history_parent);

/**
 * Executes the MCTS search algorithm.
 */
Move mcts_search(Node *root, Arena *arena, double time_limit_seconds, MCTSConfig config,
                 MCTSStats *stats, Node **out_new_root);

/**
 * Extracts the policy distribution from root (for training).
 */
void mcts_get_policy(Node *root, float *policy, float temperature, const GameState *state);

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

int get_tree_depth(Node *node);
int get_tree_node_count(Node *node);

// =============================================================================
// DIAGNOSTICS
// =============================================================================

double mcts_get_avg_root_ucb(Node *root, MCTSConfig config);
void print_mcts_stats_sorted(Node *root);

#endif // MCTS_H

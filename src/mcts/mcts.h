#ifndef MCTS_H
#define MCTS_H

// =============================================================================
// MCTS PUBLIC API
// =============================================================================
// This is the main header that clients should include.
// It re-exports all necessary types and functions.

#include "mcts_types.h"
#include "mcts_internal.h"
#include "mcts_presets.h"

// =============================================================================
// MCTS SEARCH API
// =============================================================================

/**
 * Creates the root node for the MCTS search.
 */
Node* mcts_create_root(GameState state, Arena *arena, MCTSConfig config);

/**
 * Executes the MCTS search algorithm.
 * @param root Pointer to the root node.
 * @param arena Pointer to the arena allocator.
 * @param time_limit_seconds Maximum search time.
 * @param config Search configuration.
 * @param stats Optional stats to update.
 * @param out_new_root Optional: returns chosen child for tree reuse.
 * @return The best move found.
 */
Move mcts_search(Node *root, Arena *arena, double time_limit_seconds, MCTSConfig config,
                 MCTSStats *stats, Node **out_new_root);

/**
 * Extracts the policy distribution (visit counts) from the root node.
 * @param root The root node after search.
 * @param policy Output array of size CNN_POLICY_SIZE (512).
 * @param temperature Temperature parameter (1.0 = Proportional, 0.0 = Argmax).
 * @param state Current game state (needed for context/move mapping).
 */
void mcts_get_policy(Node *root, float *policy, float temperature, const GameState *state);

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Find child node matching the given move.
 */
Node* find_child_by_move(Node *parent, const Move *move);

/**
 * Calculate tree depth.
 */
int get_tree_depth(Node *node);

/**
 * Compare two game states for equality.
 */
int states_equal(const GameState *s1, const GameState *s2);

/**
 * Calculate UCB1 score for a node.
 */
double calculate_ucb1_score(Node *child, MCTSConfig config);

// =============================================================================
// DIAGNOSTICS & DEBUG
// =============================================================================

/**
 * Calculates average UCB value of all children of the root.
 */
double mcts_get_avg_root_ucb(Node *root, MCTSConfig config);

/**
 * Prints debug info for root children, sorted by visit count.
 */
void print_mcts_stats_sorted(Node *root);

#endif // MCTS_H

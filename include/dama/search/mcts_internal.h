/**
 * mcts_internal.h - Internal MCTS Helper Functions
 * 
 * Shared helper functions used by both mcts_search.c and mcts_worker.c.
 * This eliminates code duplication between sequential and parallel MCTS paths.
 * 
 * NOT part of the public API - internal use only.
 */

#ifndef MCTS_INTERNAL_H
#define MCTS_INTERNAL_H

#include "dama/search/mcts.h"
#include "dama/search/mcts_tree.h"
#include "dama/engine/movegen.h"
#include "dama/neural/cnn.h"

// =============================================================================
// GAME RESULT HELPERS
// =============================================================================

/**
 * Determine the game result for a terminal state.
 * 
 * @param state The game state to evaluate
 * @return 1 = White Win, 2 = Black Win, 0 = Draw, -1 = Ongoing
 */
static inline int mcts_get_game_result(const GameState *state) {
    if (state->moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) return 0;
    
    MoveList ml;
    movegen_generate(state, &ml);
    
    if (ml.count == 0) {
        return (state->current_player == WHITE) ? 2 : 1;
    }
    
    return -1;
}

// =============================================================================
// TERMINAL NODE HANDLING
// =============================================================================

/**
 * Handle a terminal node: compute result and backpropagate.
 * 
 * @param leaf The terminal leaf node
 * @param config MCTS configuration
 * @param stats Optional stats to update (can be NULL)
 */
static inline void mcts_handle_terminal(Node *leaf, MCTSConfig config, MCTSStats *stats) {
    double result = 0.0;
    int res = mcts_get_game_result(&leaf->state);
    
    if (res == 1) result = (leaf->state.current_player == WHITE) ? 1.0 : 0.0;
    else if (res == 2) result = (leaf->state.current_player == BLACK) ? 1.0 : 0.0;
    else result = config.draw_score;
    
    backpropagate(leaf, result, config.use_solver);
    if (stats) stats->total_iterations++;
}

// =============================================================================
// EXPANSION HELPERS
// =============================================================================

/**
 * Expand a leaf node with CNN policy priors.
 * 
 * Thread-safe: Uses node mutex for synchronization.
 * 
 * @param leaf The leaf node to expand
 * @param arena Memory arena for allocations
 * @param config MCTS configuration
 * @param policy CNN policy output (can be NULL for vanilla MCTS)
 * @param stats Optional stats to update (can be NULL)
 * @return The expanded node (or leaf if already expanded/terminal)
 */
static inline Node* mcts_expand_with_policy(
    Node *leaf, 
    Arena *arena, 
    MCTSConfig config, 
    float *policy, 
    MCTSStats *stats
) {
    pthread_mutex_lock(&leaf->lock);
    
    if (leaf->num_children == 0 && !leaf->is_terminal) {
        MoveList legal_moves;
        movegen_generate(&leaf->state, &legal_moves);
        
        if (legal_moves.count > 0) {
            leaf->children = arena_alloc(arena, legal_moves.count * sizeof(Node*));
            
            float sum = 0.0f;
            float *filtered_policy = arena_alloc(arena, legal_moves.count * sizeof(float));
            
            for (int i = 0; i < legal_moves.count; i++) {
                int idx = cnn_move_to_index(&legal_moves.moves[i], leaf->state.current_player);
                float p = (idx >= 0 && policy) ? policy[idx] : 0.0f;
                filtered_policy[i] = p;
                sum += p;
            }
            
            for (int i = 0; i < legal_moves.count; i++) {
                if (sum > 1e-6) filtered_policy[i] /= sum;
                else filtered_policy[i] = 1.0f / legal_moves.count;
                
                GameState child_state = leaf->state;
                apply_move(&child_state, &legal_moves.moves[i]);
                
                Node *child = create_node(leaf, legal_moves.moves[i], child_state, arena, config);
                child->prior = filtered_policy[i];
                leaf->children[i] = child;
            }
            
            leaf->untried_moves.count = 0;
            atomic_thread_fence(memory_order_release);
            leaf->num_children = legal_moves.count;
            
            if (stats) {
                stats->total_expansions++;
                stats->total_policy_cached += legal_moves.count;
            }
        } else {
            leaf->is_terminal = 1;
        }
    }
    
    pthread_mutex_unlock(&leaf->lock);
    return leaf;
}

/**
 * Expand a leaf node using vanilla rollout (no CNN).
 * 
 * Thread-safe: Uses node mutex for synchronization.
 * 
 * @param leaf The leaf node to expand
 * @param arena Memory arena for allocations
 * @param config MCTS configuration
 * @param stats Optional stats to update (can be NULL)
 * @return The newly created child node
 */
static inline Node* mcts_expand_vanilla(
    Node *leaf, 
    Arena *arena, 
    MCTSConfig config, 
    MCTSStats *stats
) {
    pthread_mutex_lock(&leaf->lock);
    Node *next = leaf;
    
    if (!leaf->is_terminal) {
        next = expand_node(leaf, arena, NULL, config);
        if (stats) {
            stats->total_expansions++;
            stats->total_policy_cached++;
        }
    }
    
    pthread_mutex_unlock(&leaf->lock);
    return next;
}

#endif // MCTS_INTERNAL_H

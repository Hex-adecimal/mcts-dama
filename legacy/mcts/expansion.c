/**
 * expansion.c - MCTS Node Expansion
 * 
 * Contains: create_node, expand_node, heuristics, move comparison
 */

#include "expansion.h"
#include "../core/movegen.h"
#include "../nn/cnn.h"
#include "../nn/cnn.h"
#include <string.h>
#include <stdio.h>

double evaluate_move_heuristic(const GameState *state, const Move *move, MCTSConfig config) {
    double score = 0.0;
    int us = state->current_player;
    
    int target_idx = (move->length == 0) ? 1 : move->length;
    int from = move->path[0];
    int to   = move->path[target_idx];
    
    int row = to / 8;
    int col = to % 8;
    int from_row = from / 8;

    // 1. Capture Bonus
    if (move->length > 0) {
        score += config.weights.w_capture * move->length; 
    }

    // 2. Promotion Bonus
    if (!move->is_lady_move) {
        if (row == 0 || row == 7) { // Reaching end
            score += config.weights.w_promotion;
        }
        
        // Advancement Bonus
        int dist = (us == WHITE) ? (7 - row) : row;
        score += (7 - dist) * config.weights.w_advance;
    }

    // 3. Safety Bonus (Edges) - ONLY for regular pieces, not queens
    if (!move->is_lady_move && (col == 0 || col == 7)) {
        score += config.weights.w_edge;
    }

    // 4. Center Control
    if ((row == 3 || row == 4) && (col >= 2 && col <= 5)) {
        score += config.weights.w_center; 
    }

    // 5. Base Protection (Penalty)
    if (!move->is_lady_move) {
         if ((us == WHITE && from_row == 0) || (us == BLACK && from_row == 7)) {
             score -= config.weights.w_base;
         }
    }
    
    // 6. Queen Activity Bonus - encourage queens to move
    if (move->is_lady_move) {
        score += config.weights.w_lady_activity;
    }
    
    return score;
}

Node* create_node(Node *parent, Move move, GameState state, Arena *arena, MCTSConfig config) {
    Node *node = (Node*)arena_alloc(arena, sizeof(Node));
    
    node->state = state;
    node->move_from_parent = move;
    node->parent = parent;
    
    // If it's White's turn in the new state, it means Black just moved.
    node->player_who_just_moved = (state.current_player == WHITE) ? BLACK : WHITE;

    // Pre-allocate children array to MAX_MOVES to avoid resizing race conditions
    // This allows lock-free reading of the children pointer (though count still needs care)
    node->children = (Node**)arena_alloc(arena, MAX_MOVES * sizeof(Node*));
    node->num_children = 0;
    node->visits = 0;
    atomic_init(&node->visits, 0);
    atomic_init(&node->virtual_loss, 0);
    
    // Init Lock
    if (pthread_mutex_init(&node->lock, NULL) != 0) {
        printf("ERROR: Mutex init failed\n");
    }

    node->score = 0.0;
    node->sum_sq_score = 0.0;
    node->cached_policy = NULL; // Initialize cache to NULL

    // Generate all possible moves from this state (untried moves)
    generate_moves(&node->state, &node->untried_moves);
    node->is_terminal = (node->untried_moves.count == 0 && node->num_children == 0) ? 1 : 0; 
    
    // SOLVER INIT
    node->status = SOLVED_NONE;
    if (node->is_terminal) {
        node->status = SOLVED_LOSS; 
        
        if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
             node->status = SOLVED_DRAW;
        }
    }
    
    // HEURISTIC INIT (Progressive Bias)
    if (parent) {
        node->heuristic_score = evaluate_move_heuristic(&parent->state, &move, config);
        
        // Dynamic Threat Check (only if weight > 0)
        if (config.weights.w_threat > 0.0) {
            int dest = (move.length == 0) ? move.path[1] : move.path[move.length];
            
            if (is_square_threatened(&node->state, dest)) {
                node->heuristic_score -= config.weights.w_threat;
            }
        }
        
        // PUCT PRIOR INIT (Calculate once at creation)
        // PUCT PRIOR INIT (Calculate once at creation)
        if (config.use_puct) {
            
            // Check if Parent has a cached policy. If not, compute and normalize it.
            // This ensures:
            // 1. Efficiency: CNN runs once per node (not per child)
            // 2. Correctness: Probabilities are normalized over LEGAL moves (Fix for "Undertrained" bug)
            if (!parent->cached_policy && config.cnn_weights) {
                // Allocate in Arena
                parent->cached_policy = (float*)arena_alloc(arena, CNN_POLICY_SIZE * sizeof(float));
                
                // Run CNN with HISTORY (using ancestors)
                CNNOutput out;
                
                const GameState *hist1 = (parent->parent) ? &parent->parent->state : NULL;
                const GameState *hist2 = (parent->parent && parent->parent->parent) ? &parent->parent->parent->state : NULL;
                
                cnn_forward_with_history((CNNWeights*)config.cnn_weights, &parent->state, hist1, hist2, &out);
                
                if (0) { // Debug
                    printf("Expansion History Debug:\n");
                    if (hist1) printf("  T-1: Found\n"); else printf("  T-1: NULL (Static)\n");
                    if (hist2) printf("  T-2: Found\n"); else printf("  T-2: NULL (Static)\n");
                }
                
                // Normalize over LEGAL moves
                MoveList legal_moves;
                generate_moves(&parent->state, &legal_moves);
                
                double sum = 0.0;
                // Calculate Sum
                for (int i = 0; i < legal_moves.count; i++) {
                    int idx = cnn_move_to_index(&legal_moves.moves[i], parent->state.current_player);
                    if (idx >= 0 && idx < CNN_POLICY_SIZE) {
                        sum += out.policy[idx];
                    }
                }
                
                // Store Normalized
                for (int i = 0; i < CNN_POLICY_SIZE; i++) parent->cached_policy[i] = 0.0f; // Clear
                
                if (sum < 1e-9) sum = 1.0; // Avoid division by zero
                double scale = 1.0 / sum;
                
                for (int i = 0; i < legal_moves.count; i++) {
                     int idx = cnn_move_to_index(&legal_moves.moves[i], parent->state.current_player);
                     if (idx >= 0 && idx < CNN_POLICY_SIZE) {
                         parent->cached_policy[idx] = out.policy[idx] * scale;
                     }
                }
            }
            
            // Assign Prior from Parent's Cache
            if (parent->cached_policy) {
                int idx = cnn_move_to_index(&move, parent->state.current_player);
                if (idx >= 0 && idx < CNN_POLICY_SIZE) {
                    node->prior = parent->cached_policy[idx];
                } else {
                    node->prior = 0.0f;
                }
            } else if (config.nn_weights) {
                 // Legacy MLP removed - use default prior
                 node->prior = 1.0f;
            } else {
                 node->prior = 1.0f;
            }
        } else {
            node->prior = 0.0f;
        }

        // LOOP DETECTION: Check if this state repeats an ancestor
        Node *ancestor = parent;
        while (ancestor) {
            if (ancestor->state.hash == node->state.hash) {
                // Loop detected! Treat as a bad outcome to discourage shuffling.
                node->is_terminal = 1;
                // We mark it as SOLVED_DRAW, but give a massive heuristic penalty
                node->status = SOLVED_DRAW; 
                node->heuristic_score -= 50000.0; // Huge penalty
                node->score = -1.0; // Initialize with loss-like score
                break;
            }
            ancestor = ancestor->parent;
        }

    } else {
        node->heuristic_score = 0.0;
        node->prior = 0.0f;
    }

    return node;
}

int moves_equal(const Move *m1, const Move *m2) {
    if (m1->length != m2->length) return 0;
    
    int limit = (m1->length == 0) ? 1 : m1->length;

    for (int i = 0; i <= limit; i++) {
        if (m1->path[i] != m2->path[i]) return 0;
    }
    return 1;
}

Node* find_child_by_move(Node *parent, const Move *move) {
    if (!parent || !move) return NULL;
    
    for (int i = 0; i < parent->num_children; i++) {
        if (moves_equal(&parent->children[i]->move_from_parent, move)) {
            return parent->children[i];
        }
    }
    return NULL;
}

Node* expand_node(Node *node, Arena *arena, TranspositionTable *tt, MCTSConfig config) {
    if (node->untried_moves.count == 0) return node;

    // Pop the last move from untried_moves (efficient)
    int idx = node->untried_moves.count - 1;
    Move move_to_try = node->untried_moves.moves[idx];
    node->untried_moves.count--;

    // Apply the move to get the new state
    GameState next_state = node->state;
    apply_move(&next_state, &move_to_try);

    // Create the child node
    Node *child = create_node(node, move_to_try, next_state, arena, config);

    // Handle Transposition Table
    if (tt) {
        Node *match = tt_lookup(tt, &next_state);
        if (match) {
            // Found in TT: Warm-start this node with stats from the transposition!
            child->visits = match->visits;
            child->score  = match->score;
            child->sum_sq_score = match->sum_sq_score;
            child->status = match->status;
        }
        
        tt_insert(tt, child);
    }

    // Insert child into pre-allocated array
    // Lockless read of 'children' pointer is safe because it's constant.
    // Lock on 'node->lock' (held by caller) protects 'num_children' update.
    
    if (node->num_children >= MAX_MOVES) {
        printf("ERROR: Exceeded MAX_MOVES in expand_node\n");
        return node; // Should not happen if movegen is correct
    }

    node->children[node->num_children] = child;
    
    // Memory barrier to ensure child pointer is visible before count increment
    atomic_thread_fence(memory_order_release);
    
    node->num_children++;

    return child;
}

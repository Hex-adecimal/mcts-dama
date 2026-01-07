/**
 * mcts_tree.c - MCTS Tree Operations
 * 
 * Node creation, expansion, and backpropagation.
 * Selection algorithms are in mcts_selection.c.
 */

#include "dama/search/mcts_tree.h"
#include "dama/search/mcts_types.h"
#include "dama/engine/movegen.h"
#include "dama/neural/cnn.h"
#include "dama/common/params.h"
#include "dama/common/debug.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

// =============================================================================
// HEURISTIC EVALUATION
// =============================================================================

double evaluate_move_heuristic(const GameState *state, const Move *move, MCTSConfig config) {
    double score = 0.0;
    int us = state->current_player;
    
    int target_idx = (move->length == 0) ? 1 : move->length;
    int from = move->path[0];
    int to = move->path[target_idx];
    int row = to / 8;
    int col = to % 8;
    int from_row = from / 8;

    // Capture bonus
    if (move->length > 0) score += config.weights.w_capture * move->length;

    // Promotion bonus
    if (!move->is_lady_move) {
        if (row == 0 || row == 7) score += config.weights.w_promotion;
        int dist = (us == WHITE) ? (7 - row) : row;
        score += (7 - dist) * config.weights.w_advance;
    }

    // Edge safety (pieces only)
    if (!move->is_lady_move && (col == 0 || col == 7)) score += config.weights.w_edge;

    // Center control
    if ((row == 3 || row == 4) && (col >= 2 && col <= 5)) score += config.weights.w_center;

    // Base protection penalty
    if (!move->is_lady_move) {
        if ((us == WHITE && from_row == 0) || (us == BLACK && from_row == 7)) {
            score -= config.weights.w_base;
        }
    }
    
    // Queen activity
    if (move->is_lady_move) score += config.weights.w_lady_activity;
    
    return score;
}

// =============================================================================
// NODE CREATION
// =============================================================================

Node* create_node(Node *parent, Move move, GameState state, Arena *arena, MCTSConfig config) {
    DBG_NOT_NULL(arena);
    Node *node = (Node*)arena_alloc(arena, sizeof(Node));
    if (!node) return NULL;
    
    memset(node, 0, sizeof(Node));
    node->state = state;
    node->move_from_parent = move;
    node->parent = parent;
    node->player_who_just_moved = (state.current_player == WHITE) ? BLACK : WHITE;

    node->children = (Node**)arena_alloc(arena, MAX_MOVES * sizeof(Node*));
    node->num_children = 0;
    atomic_init(&node->visits, 0);
    atomic_init(&node->virtual_loss, 0);
    
    if (pthread_mutex_init(&node->lock, NULL) != 0) {
        printf("ERROR: Mutex init failed\n");
    }

    node->score = 0.0;
    node->sum_sq_score = 0.0;
    
    node->heuristic_score = evaluate_move_heuristic(&node->state, &node->move_from_parent, config);
    
    movegen_generate(&node->state, &node->untried_moves);
    node->is_terminal = (node->untried_moves.count == 0 && node->num_children == 0) ? 1 : 0;
    
    // Solver init
    node->status = SOLVED_NONE;
    
    if (config.verbose) printf("Creation End %p Heur=%.2f\n", node, node->heuristic_score);
    if (node->is_terminal) {
        node->status = SOLVED_LOSS;
        if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
            node->status = SOLVED_DRAW;
        }
    }
    
    // Heuristic & PUCT init
    if (parent) {
        node->heuristic_score = evaluate_move_heuristic(&parent->state, &move, config);
        
        if (config.weights.w_threat > 0.0) {
            int dest = (move.length == 0) ? move.path[1] : move.path[move.length];
            if (movegen_is_square_threatened(&node->state, dest)) {
                node->heuristic_score -= config.weights.w_threat;
            }
        }
        
        node->prior = 0.0f; // Assigned by caller (perform_expansion)


        // Loop detection
        Node *ancestor = parent;
        while (ancestor) {
            if (ancestor->state.hash == node->state.hash) {
                node->is_terminal = 1;
                node->status = SOLVED_DRAW;
                node->heuristic_score -= 50000.0;
                node->score = -1.0;
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

// =============================================================================
// MOVE COMPARISON
// =============================================================================

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

// =============================================================================
// EXPANSION
// =============================================================================

Node* expand_node(Node *node, Arena *arena, TranspositionTable *tt, MCTSConfig config, MCTSStats *stats) {
    DBG_NOT_NULL(node);
    DBG_NOT_NULL(arena);
    if (node->untried_moves.count == 0) return node;

    int idx = node->untried_moves.count - 1;
    Move move_to_try = node->untried_moves.moves[idx];
    node->untried_moves.count--;

    GameState next_state = node->state;
    apply_move(&next_state, &move_to_try);

    Node *child = create_node(node, move_to_try, next_state, arena, config);

    if (tt) {
        Node *match = tt_lookup(tt, &next_state);
        if (match) {
            child->visits = match->visits;
            child->score = match->score;
            child->sum_sq_score = match->sum_sq_score;
            child->status = match->status;
            if (stats) stats->tt_hits++;
        } else {
            if (stats) stats->tt_misses++;
        }
        tt_insert(tt, child);
    }

    if (node->num_children >= MAX_MOVES) {
        printf("ERROR: Exceeded MAX_MOVES in expand_node\n");
        return node;
    }

    node->children[node->num_children] = child;
    atomic_thread_fence(memory_order_release);
    node->num_children++;

    return child;
}

// =============================================================================
// BACKPROPAGATION
// =============================================================================

/**
 * Update solver status (Proven Win/Loss/Draw) for a node based on its children.
 * 
 * Propagates Minimax values up the tree:
 * - If any child is a LOSS for opponent -> WIN for us.
 * - If ALL children are WINS for opponent -> LOSS for us.
 * - If we can force a DRAW -> DRAW status.
 */
void update_solver_status(Node *node) {
    if (node->status != SOLVED_NONE) return;
    
    // Can we win? (any child is loss for opponent)
    for (int i = 0; i < node->num_children; i++) {
        if (node->children[i]->status == SOLVED_LOSS) {
            node->status = SOLVED_WIN;
            return;
        }
    }

    // Are we forced to lose? (fully expanded, all children are wins for opponent)
    if (node->untried_moves.count == 0) {
        int all_win = 1;
        for (int i = 0; i < node->num_children; i++) {
            if (node->children[i]->status != SOLVED_WIN) {
                all_win = 0;
                break;
            }
        }
        if (all_win && node->num_children > 0) {
            node->status = SOLVED_LOSS;
        }
    }
}

void backpropagate(Node *node, double result, int use_solver) {
    DBG_NOT_NULL(node);
    Node *child = NULL;
    
    while (node != NULL) {
        atomic_fetch_sub(&node->virtual_loss, 1);
        atomic_fetch_add(&node->visits, 1);
        
        pthread_mutex_lock(&node->lock);
        node->score += result;
        node->sum_sq_score += (result * result);
        
        if (use_solver && (child == NULL || child->status != SOLVED_NONE)) {
            update_solver_status(node);
        }
        pthread_mutex_unlock(&node->lock);

        result = 1.0 - result;
        child = node;
        node = node->parent;
    }
}

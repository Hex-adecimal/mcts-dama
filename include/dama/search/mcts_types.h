/**
 * mcts_types.h - MCTS Core Types
 * 
 * Contains: Node struct, SolverStatus
 * Config, Stats, Presets are in mcts_config.h
 * TranspositionTable is in mcts_tt.h
 */

#ifndef MCTS_TYPES_H
#define MCTS_TYPES_H

#include "dama/engine/game.h"
#include "dama/search/mcts_arena.h"
#include "dama/search/mcts_config.h"
#include "dama/search/mcts_tt.h"
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>

// =============================================================================
// SOLVER STATUS
// =============================================================================

typedef enum {
    SOLVED_NONE = 0,
    SOLVED_WIN = 1,
    SOLVED_LOSS = -1,
    SOLVED_DRAW = 2
} SolverStatus;

// =============================================================================
// MCTS NODE
// =============================================================================

typedef struct Node {
    GameState state;
    Move move_from_parent;
    int player_who_just_moved;
    
    struct Node *parent;
    struct Node **children;
    int num_children;
    
    MoveList untried_moves;
    int is_terminal;
    
    _Atomic int visits;
    _Atomic int virtual_loss;
    
    double score;
    double sum_sq_score;
    double heuristic_score;
    float prior;
    float *cached_policy;
    
    pthread_mutex_t lock;
    int8_t status;  // SolverStatus
} Node;

// =============================================================================
// TRANSPOSITION TABLE LOOKUP/INSERT (need Node definition)
// =============================================================================

static inline Node* tt_lookup(TranspositionTable *tt, const GameState *state) {
    if (!tt) return NULL;
    size_t idx = state->hash & tt->mask;
    
    pthread_mutex_lock(&tt->lock);
    Node *node = tt->buckets[idx];
    if (node && node->state.hash == state->hash && states_equal(&node->state, state)) {
        pthread_mutex_unlock(&tt->lock);
        return node;
    }
    pthread_mutex_unlock(&tt->lock);
    return NULL;
}

static inline void tt_insert(TranspositionTable *tt, Node *node) {
    if (!tt) return;
    size_t idx = node->state.hash & tt->mask;
    
    pthread_mutex_lock(&tt->lock);
    if (tt->buckets[idx] != NULL) tt->collisions++;
    tt->buckets[idx] = node;
    tt->count++;
    pthread_mutex_unlock(&tt->lock);
}

#endif // MCTS_TYPES_H

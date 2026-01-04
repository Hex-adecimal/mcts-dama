/**
 * mcts_tt.h - MCTS Transposition Table
 * 
 * Extracted from mcts_types.h for better modularity.
 * Contains: TranspositionTable struct and operations
 */

#ifndef MCTS_TT_H
#define MCTS_TT_H

#include <stddef.h>
#include <stdlib.h>
#include <pthread.h>

// Forward declaration (Node is defined in mcts_types.h)
struct Node;

// =============================================================================
// TRANSPOSITION TABLE
// =============================================================================

typedef struct {
    struct Node **buckets;
    size_t size;
    size_t mask;
    size_t count;
    size_t collisions;
    pthread_mutex_t lock;
} TranspositionTable;

// Helper to compare game states (needs GameState from game.h)
#include "dama/engine/game.h"

static inline int states_equal(const GameState *s1, const GameState *s2) {
    return s1->piece[WHITE][PAWN] == s2->piece[WHITE][PAWN] &&
           s1->piece[WHITE][LADY] == s2->piece[WHITE][LADY] &&
           s1->piece[BLACK][PAWN] == s2->piece[BLACK][PAWN] &&
           s1->piece[BLACK][LADY] == s2->piece[BLACK][LADY] &&
           s1->current_player == s2->current_player &&
           s1->moves_without_captures == s2->moves_without_captures;
}

static inline TranspositionTable* tt_create(size_t size) {
    TranspositionTable *tt = malloc(sizeof(TranspositionTable));
    if (!tt) return NULL;
    tt->size = size;
    tt->mask = size - 1;
    tt->count = 0;
    tt->collisions = 0;
    tt->buckets = calloc(size, sizeof(struct Node*));
    if (!tt->buckets) {
        free(tt);
        return NULL;
    }
    pthread_mutex_init(&tt->lock, NULL);
    return tt;
}

static inline void tt_free(TranspositionTable *tt) {
    if (tt) {
        pthread_mutex_destroy(&tt->lock);
        free(tt->buckets);
        free(tt);
    }
}

#endif // MCTS_TT_H

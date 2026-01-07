/**
 * mcts_types.h - MCTS Core Types and Memory Management
 * 
 * Contains:
 * - Arena allocator (Thread-safe bump allocator)
 * - TranspositionTable
 * - Node struct and SolverStatus
 * 
 * Config, Stats, Presets are in mcts_config.h
 */

#ifndef MCTS_TYPES_H
#define MCTS_TYPES_H

#include "dama/engine/game.h"
#include "dama/common/logging.h"
#include "dama/search/mcts_config.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <pthread.h>

// =============================================================================
// ARENA ALLOCATOR
// =============================================================================

typedef struct {
    unsigned char *buffer;
    size_t size;
    size_t offset;
    pthread_mutex_t lock;
} Arena;

/**
 * Initialize arena allocator.
 * @return 0 on success, -1 on failure (malloc failed)
 */
static inline int arena_init(Arena *a, size_t total_size) {
    a->buffer = malloc(total_size);
    if (!a->buffer) {
        log_error("[Arena] Malloc failed for size %zu", total_size);
        return -1;
    }
    a->size = total_size;
    a->offset = 0;
    pthread_mutex_init(&a->lock, NULL);
    return 0;
}

/**
 * Allocate memory from arena.
 * @return Pointer to allocated memory, or NULL if out of memory
 */
static inline void* arena_alloc(Arena *a, size_t bytes) {
    pthread_mutex_lock(&a->lock);
    uintptr_t current_ptr = (uintptr_t)(a->buffer + a->offset);
    uintptr_t padding = (8 - (current_ptr % 8)) % 8;
    
    if (a->offset + padding + bytes > a->size) {
        pthread_mutex_unlock(&a->lock);
        log_error("[Arena] Out of Memory! (Size: %zu, Requested: %zu)", a->size, bytes);
        return NULL;
    }
    
    void *ptr = a->buffer + a->offset + padding;
    a->offset += padding + bytes;
    pthread_mutex_unlock(&a->lock);
    return ptr;
}

static inline void arena_reset(Arena *a) {
    pthread_mutex_lock(&a->lock);
    a->offset = 0;
    pthread_mutex_unlock(&a->lock);
}

static inline void arena_free(Arena *a) {
    pthread_mutex_destroy(&a->lock);
    free(a->buffer);
    a->buffer = NULL;
}

// =============================================================================
// TRANSPOSITION TABLE
// =============================================================================

// Forward declaration for TT (Node is defined below)
struct Node;

typedef struct {
    struct Node **buckets;
    size_t size;
    size_t mask;
    size_t count;
    size_t collisions;
    pthread_mutex_t lock;
} TranspositionTable;

// Helper to compare game states
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

static inline void tt_reset(TranspositionTable *tt) {
    if (tt) {
        pthread_mutex_lock(&tt->lock);
        memset(tt->buckets, 0, tt->size * sizeof(struct Node*));
        tt->count = 0;
        tt->collisions = 0;
        pthread_mutex_unlock(&tt->lock);
    }
}

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
    Node *existing = tt->buckets[idx];
    
    // Quality-based replacement: only replace if new node has more visits
    if (existing && atomic_load(&existing->visits) > atomic_load(&node->visits)) {
        pthread_mutex_unlock(&tt->lock);
        return;  // Keep existing higher-quality entry
    }
    
    if (tt->buckets[idx] != NULL) tt->collisions++;
    tt->buckets[idx] = node;
    tt->count++;
    pthread_mutex_unlock(&tt->lock);
}

#endif // MCTS_TYPES_H

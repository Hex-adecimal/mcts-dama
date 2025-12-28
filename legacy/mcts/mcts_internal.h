#ifndef MCTS_INTERNAL_H
#define MCTS_INTERNAL_H

#include "../core/game.h"
#include <pthread.h>
#include <stddef.h>

/**
 * Simple Arena Allocator.
 * Allocates MCTS nodes in a contiguous memory block, avoiding fragmentation.
 */
typedef struct {
    unsigned char *buffer;
    size_t size;
    size_t offset;
    pthread_mutex_t lock;
} Arena;

/**
 * Transposition Table: maps Zobrist Hash -> Node pointer.
 */
typedef struct {
    struct Node **buckets;
    size_t size;
    size_t mask;
    size_t count;
    size_t collisions;
    pthread_mutex_t lock;
} TranspositionTable;

// Forward declaration of Node (defined in mcts_types.h)
struct Node;

// Arena API
void arena_init(Arena *a, size_t total_size);
void* arena_alloc(Arena *a, size_t bytes);
void arena_reset(Arena *a);
void arena_free(Arena *a);

// Transposition Table API
TranspositionTable* tt_create(size_t size);
void tt_free(TranspositionTable *tt);
struct Node* tt_lookup(TranspositionTable *tt, const GameState *state);
void tt_insert(TranspositionTable *tt, struct Node *node);

// Utility
int states_equal(const GameState *s1, const GameState *s2);

#endif // MCTS_INTERNAL_H

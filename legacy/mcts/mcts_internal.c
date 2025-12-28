/**
 * mcts_internal.c - Support structures for MCTS (Arena & Transposition Table)
 */

#include "mcts_internal.h"
#include "mcts_types.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

// =============================================================================
// ARENA ALLOCATOR
// =============================================================================

void arena_init(Arena *a, size_t total_size) {
    a->buffer = malloc(total_size);
    if (!a->buffer) { 
        fprintf(stderr, "FATAL: Malloc failed for Arena of size %zu\n", total_size); 
        exit(1); 
    }
    a->size = total_size;
    a->offset = 0; 
    if (pthread_mutex_init(&a->lock, NULL) != 0) {
        fprintf(stderr, "Warning: Arena Mutex init failed\n");
    }
}

void* arena_alloc(Arena *a, size_t bytes) {
    pthread_mutex_lock(&a->lock);
    
    // Align pointer to 8 bytes
    uintptr_t current_ptr = (uintptr_t)(a->buffer + a->offset);
    uintptr_t padding = (8 - (current_ptr % 8)) % 8; 
    
    if (a->offset + padding + bytes > a->size) {
        pthread_mutex_unlock(&a->lock);
        fprintf(stderr, "FATAL: Arena Out of Memory! (Size: %zu)\n", a->size);
        exit(1);
    }
    
    void *ptr = a->buffer + a->offset + padding;
    a->offset += padding + bytes;
    
    pthread_mutex_unlock(&a->lock);
    return ptr;
}

void arena_reset(Arena *a) { 
    pthread_mutex_lock(&a->lock);
    a->offset = 0; 
    pthread_mutex_unlock(&a->lock);
}

void arena_free(Arena *a) { 
    pthread_mutex_destroy(&a->lock);
    free(a->buffer); 
}

// =============================================================================
// TRANSPOSITION TABLE
// =============================================================================

TranspositionTable* tt_create(size_t size) {
    TranspositionTable *tt = malloc(sizeof(TranspositionTable));
    tt->size = size;
    tt->mask = size - 1;
    tt->count = 0;
    tt->collisions = 0;
    tt->buckets = calloc(size, sizeof(struct Node*)); 
    
    if (pthread_mutex_init(&tt->lock, NULL) != 0) {
        printf("Warning: TT Mutex init failed\n");
    }
    
    return tt;
}

void tt_free(TranspositionTable *tt) {
    if (tt) {
        pthread_mutex_destroy(&tt->lock);
        free(tt->buckets);
        free(tt);
    }
}

int states_equal(const GameState *s1, const GameState *s2) {
    if (s1->white_pieces != s2->white_pieces) return 0;
    if (s1->black_pieces != s2->black_pieces) return 0;
    if (s1->white_ladies != s2->white_ladies) return 0;
    if (s1->black_ladies != s2->black_ladies) return 0;
    if (s1->current_player != s2->current_player) return 0;
    if (s1->moves_without_captures != s2->moves_without_captures) return 0;
    return 1;
}

struct Node* tt_lookup(TranspositionTable *tt, const GameState *state) {
    if (!tt) return NULL;
    size_t idx = state->hash & tt->mask;
    
    pthread_mutex_lock(&tt->lock);
    struct Node *node = tt->buckets[idx];
    if (node && node->state.hash == state->hash) {
        if (states_equal(&node->state, state)) {
            pthread_mutex_unlock(&tt->lock);
            return node;
        }
    }
    pthread_mutex_unlock(&tt->lock);
    return NULL; 
}

void tt_insert(TranspositionTable *tt, struct Node *node) {
    if (!tt) return;
    size_t idx = node->state.hash & tt->mask;
    
    pthread_mutex_lock(&tt->lock);
    if (tt->buckets[idx] != NULL) tt->collisions++;
    tt->buckets[idx] = node;
    tt->count++;
    pthread_mutex_unlock(&tt->lock);
}

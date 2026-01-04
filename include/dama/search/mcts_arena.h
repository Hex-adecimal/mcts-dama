/**
 * mcts_arena.h - Arena Memory Allocator for MCTS
 * 
 * Thread-safe bump allocator for efficient MCTS node allocation.
 * Returns NULL on failure instead of calling exit().
 */

#ifndef MCTS_ARENA_H
#define MCTS_ARENA_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include "dama/common/logging.h"

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

#endif // MCTS_ARENA_H

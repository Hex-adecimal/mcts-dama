/**
 * mcts_worker.h - MCTS Multi-threaded Worker Infrastructure
 * 
 * Header for worker thread types and functions.
 */

#ifndef MCTS_WORKER_H
#define MCTS_WORKER_H

#include "dama/search/mcts_types.h"
#include <pthread.h>

// =============================================================================
// ASYNC BATCHING INFRASTRUCTURE
// =============================================================================

/**
 * Request structure for async CNN inference batching.
 */
typedef struct {
    Node *node;
    float *policy_out;
    float *value_out;
    int ready;
    pthread_cond_t cond;
} InferenceRequest;

/**
 * Queue for batching inference requests across worker threads.
 */
typedef struct {
    GameState states[MCTS_BATCH_SIZE];
    InferenceRequest *requests[MCTS_BATCH_SIZE];
    int count;
    
    pthread_mutex_t lock;
    pthread_cond_t cond_batch_ready;
    
    volatile int shutdown;
} InferenceQueue;

/**
 * Arguments passed to each worker thread.
 */
typedef struct {
    Node *root;
    Arena *arena;
    MCTSConfig config;
    InferenceQueue *queue;
    int thread_id;
    TranspositionTable *tt;
    MCTSStats *local_stats;
} WorkerArgs;

// =============================================================================
// WORKER THREAD API
// =============================================================================

/**
 * MCTS Worker Thread function.
 * 
 * Performs Selection, Expansion, and Backpropagation.
 * For CNN evaluation, coordinates with other threads to form efficient batches.
 */
void *mcts_worker(void *arg);

#endif // MCTS_WORKER_H

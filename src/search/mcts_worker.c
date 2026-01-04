/**
 * mcts_worker.c - MCTS Multi-threaded Worker Infrastructure
 * 
 * Extracted from mcts_search.c for better modularity.
 * Contains: InferenceQueue, WorkerArgs, mcts_worker()
 * 
 * Uses shared helpers from mcts_internal.h to avoid code duplication.
 */

#include "dama/search/mcts.h"
#include "dama/search/mcts_tree.h"
#include "dama/search/mcts_internal.h"
#include "dama/neural/cnn.h"
#include "dama/common/params.h"
#include "dama/engine/movegen.h"
#include <string.h>
#include <sched.h>

// Forward declarations
double simulate_rollout(Node *node, MCTSConfig config);
extern Node* select_promising_node(Node *root, MCTSConfig config);
extern void backpropagate(Node *node, double result, int use_solver);
extern Node* expand_node(Node *node, Arena *arena, TranspositionTable *tt, MCTSConfig config);
extern Node* create_node(Node *parent, Move move, GameState state, Arena *arena, MCTSConfig config);

// =============================================================================
// ASYNC BATCHING INFRASTRUCTURE
// =============================================================================

typedef struct {
    Node *node;
    float *policy_out;
    float *value_out;
    int ready;
    pthread_cond_t cond;
} InferenceRequest;

typedef struct {
    GameState states[MCTS_BATCH_SIZE];
    InferenceRequest *requests[MCTS_BATCH_SIZE];
    int count;
    
    pthread_mutex_t lock;
    pthread_cond_t cond_batch_ready;
    
    volatile int shutdown;
} InferenceQueue;

typedef struct {
    Node *root;
    Arena *arena;
    MCTSConfig config;
    InferenceQueue *queue;
    int thread_id;
    MCTSStats *local_stats;
} WorkerArgs;

// =============================================================================
// WORKER EXPANSION WRAPPER
// =============================================================================

// Wrapper that dispatches to the appropriate shared helper based on config
static Node* perform_expansion_worker(Node *leaf, Arena *arena, MCTSConfig config, float *policy, MCTSStats *stats) {
    if (config.cnn_weights) {
        return mcts_expand_with_policy(leaf, arena, config, policy, stats);
    } else {
        return mcts_expand_vanilla(leaf, arena, config, stats);
    }
}

// =============================================================================
// WORKER THREAD FUNCTION
// =============================================================================

/**
 * MCTS Worker Thread.
 * 
 * Performs Selection, Expansion, and Backpropagation.
 * For Evaluation, it coordinates with other threads to form efficient batches
 * for the CNN Inference, using a condition variable/queue mechanism.
 */
void *mcts_worker(void *arg) {
    WorkerArgs *args = (WorkerArgs*)arg;
    Node *root = args->root;
    InferenceQueue *queue = args->queue;
    MCTSConfig config = args->config;
    
    while (!queue->shutdown) {
        // 1. Selection (with Virtual Loss)
        Node *leaf = select_promising_node(root, config);
        
        // Check for terminal state - use shared helper from mcts_internal.h
        if (leaf->is_terminal) {
            mcts_handle_terminal(leaf, config, args->local_stats);
             continue;
        }

        // 2. Evaluation (Neural Net or Vanilla)
        float value;
        float policy[CNN_POLICY_SIZE];
        
        if (config.cnn_weights) {
            // --- ASYNC BATCHING ---
            InferenceRequest req;
            req.node = leaf;
            req.policy_out = policy;
            req.value_out = &value;
            req.ready = 0;
            pthread_cond_init(&req.cond, NULL);
            
            pthread_mutex_lock(&queue->lock);
            
            while (queue->count >= MCTS_BATCH_SIZE && !queue->shutdown) {
                pthread_cond_signal(&queue->cond_batch_ready);
                pthread_mutex_unlock(&queue->lock);
                sched_yield(); 
                pthread_mutex_lock(&queue->lock);
            }
            
            if (queue->shutdown) {
                pthread_mutex_unlock(&queue->lock);
                pthread_cond_destroy(&req.cond);
                break; 
            }

            // Enqueue
            queue->states[queue->count] = leaf->state;
            queue->requests[queue->count] = &req;
            queue->count++;
            
            if (queue->count >= MCTS_BATCH_SIZE) {
                pthread_cond_signal(&queue->cond_batch_ready);
            }
            
            // Wait for result
            while (!req.ready && !queue->shutdown) {
                 pthread_cond_wait(&req.cond, &queue->lock);
            }
            
            pthread_mutex_unlock(&queue->lock);
            pthread_cond_destroy(&req.cond);
            
            if (queue->shutdown) break;
            
        } else {
            // Vanilla Rollout
            value = simulate_rollout(leaf, config);
        }

        // 3. Expansion
        Node *next_leaf = perform_expansion_worker(leaf, args->arena, config, (config.cnn_weights ? policy : NULL), args->local_stats);
        
        // 4. Backpropagation
        backpropagate(next_leaf, value, config.use_solver);
        if (args->local_stats) args->local_stats->total_iterations++;
    }
    return NULL;
}

// Getters for internal types (used by mcts_search.c)
size_t mcts_worker_get_queue_size(void) {
    return sizeof(InferenceQueue);
}

size_t mcts_worker_get_args_size(void) {
    return sizeof(WorkerArgs);
}

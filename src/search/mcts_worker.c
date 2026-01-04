/**
 * mcts_worker.c - MCTS Multi-threaded Worker Infrastructure
 * 
 * Extracted from mcts_search.c for better modularity.
 * Contains: InferenceQueue, WorkerArgs, mcts_worker()
 */

#include "dama/search/mcts.h"
#include "dama/search/mcts_tree.h"
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
// HELPER FUNCTIONS (copied from mcts_search.c)
// =============================================================================

// Helper to determine game result for terminal nodes
static int get_game_result_worker(const GameState *state) {
    if (state->moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) return 0;
    
    MoveList ml;
    generate_moves(state, &ml);
    
    if (ml.count == 0) {
        return (state->current_player == WHITE) ? 2 : 1;
    }
    
    return -1;
}

static void solve_terminal_node_worker(Node *leaf, MCTSConfig config, MCTSStats *stats) {
    double result = 0.0;
    int res = get_game_result_worker(&leaf->state);
    if (res == 1) result = (leaf->state.current_player == WHITE) ? 1.0 : 0.0;
    else if (res == 2) result = (leaf->state.current_player == BLACK) ? 1.0 : 0.0;
    else result = config.draw_score;
    
    backpropagate(leaf, result, config.use_solver);
    if (stats) stats->total_iterations++;
}

static Node* perform_expansion_worker(Node *leaf, Arena *arena, MCTSConfig config, float *policy, MCTSStats *stats) {
    if (config.cnn_weights) {
        pthread_mutex_lock(&leaf->lock);
        if (leaf->num_children == 0 && !leaf->is_terminal) {
            MoveList legal_moves;
            generate_moves(&leaf->state, &legal_moves);
            
            if (legal_moves.count > 0) {
                leaf->children = arena_alloc(arena, legal_moves.count * sizeof(Node*));
                
                float sum = 0.0f;
                float *filtered_policy = arena_alloc(arena, legal_moves.count * sizeof(float));
                
                for (int i=0; i<legal_moves.count; i++) {
                    int idx = cnn_move_to_index(&legal_moves.moves[i], leaf->state.current_player);
                    float p = (idx >= 0 && policy) ? policy[idx] : 0.0f;
                    filtered_policy[i] = p;
                    sum += p;
                }
                
                for (int i=0; i<legal_moves.count; i++) {
                    if (sum > 1e-6) filtered_policy[i] /= sum;
                    else filtered_policy[i] = 1.0f / legal_moves.count;
                    GameState child_state = leaf->state;
                    apply_move(&child_state, &legal_moves.moves[i]);
                    
                    Node *child = create_node(leaf, legal_moves.moves[i], child_state, arena, config);
                    
                    child->prior = filtered_policy[i];
                    leaf->children[i] = child;
                }
                
                leaf->untried_moves.count = 0;
                atomic_thread_fence(memory_order_release);
                leaf->num_children = legal_moves.count;
                
                if (stats) {
                    stats->total_expansions++;
                    stats->total_policy_cached += legal_moves.count;
                }

            } else {
                leaf->is_terminal = 1;
            }
        }
        pthread_mutex_unlock(&leaf->lock);
        return leaf;
    } else {
        pthread_mutex_lock(&leaf->lock);
        Node *next = leaf;
        if (!leaf->is_terminal) {
            next = expand_node(leaf, arena, NULL, config);
            if (stats) {
                stats->total_expansions++;
                stats->total_policy_cached++;
            }
        }
        pthread_mutex_unlock(&leaf->lock);
        return next;
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
        
        // Check for terminal state
        if (leaf->is_terminal) {
            solve_terminal_node_worker(leaf, config, args->local_stats);
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

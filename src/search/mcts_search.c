/**
 * mcts_search.c - MCTS Main Search Algorithm
 * 
 * Contains: mcts_search, mcts_step_sequential, should_exit_early
 * Worker threads are in mcts_worker.c
 */

#include "dama/search/mcts.h"
#include "dama/search/mcts_tree.h"
#include "dama/search/mcts_internal.h"
#include "dama/search/mcts_worker.h"
#include "dama/neural/cnn.h"
#include "dama/common/params.h"
#include "dama/common/debug.h"
#include "dama/engine/movegen.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <sched.h>


// External functions from mcts_utils.c
extern int get_tree_depth(const Node *node);
extern int get_tree_node_count(const Node *node);

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Helper to determine game result
// Returns: 1 = White Win, 2 = Black Win, 0 = Draw, -1 = Ongoing
static int get_game_result(const GameState *state) {
    DBG_NOT_NULL(state);
    if (state->moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) return 0; // Draw
    
    MoveList ml;
    movegen_generate(state, &ml);
    
    if (ml.count == 0) {
        // Current player has no moves -> Loses
        return (state->current_player == WHITE) ? 2 : 1;
    }
    
    return -1; // Ongoing
}

// Early Exit Check: Returns 1 if the best move cannot be overtaken
static int should_exit_early(Node *root, int max_nodes) {
    if (!root || root->num_children < 2) return 0;
    
    int best_visits = -1;
    int second_best_visits = -1;
    
    for (int i = 0; i < root->num_children; i++) {
        int v = root->children[i]->visits;
        if (v > best_visits) {
            second_best_visits = best_visits;
            best_visits = v;
        } else if (v > second_best_visits) {
            second_best_visits = v;
        }
    }
    
    int remaining = max_nodes - root->visits;
    if (best_visits > second_best_visits + remaining) {
        return 1; // Early exit triggered
    }
    
    return 0;
}

// --- Common Refactored Steps ---

static inline Node* perform_selection(Node *root, MCTSConfig config) {
    return select_promising_node(root, config);
}

static inline void solve_terminal_node(Node *leaf, MCTSConfig config, MCTSStats *stats) {
    double result = 0.0;
    int res = get_game_result(&leaf->state);
    if (res == 1) result = (leaf->state.current_player == WHITE) ? 1.0 : 0.0;
    else if (res == 2) result = (leaf->state.current_player == BLACK) ? 1.0 : 0.0;
    else result = config.draw_score;
    
    backpropagate(leaf, result, config.use_solver);
    if (stats) stats->total_iterations++;
}

static inline Node* perform_expansion(Node *leaf, Arena *arena, TranspositionTable *tt, MCTSConfig config, float *policy, MCTSStats *stats) {
    if (config.cnn_weights) {
        pthread_mutex_lock(&leaf->lock);
        if (leaf->num_children == 0 && !leaf->is_terminal) {
            MoveList legal_moves;
            movegen_generate(&leaf->state, &legal_moves);
            
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
                    
                    Node *child = NULL;
                    if (tt) {
                        child = tt_lookup(tt, &child_state);
                        if (child) {
                            if (stats) stats->tt_hits++;
                        }
                    }
                    
                    if (!child) {
                        child = create_node(leaf, legal_moves.moves[i], child_state, arena, config);
                        if (tt) {
                            tt_insert(tt, child);
                            if (stats) stats->tt_misses++;
                        }
                    }
                    
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
            next = expand_node(leaf, arena, tt, config, stats);
            if (stats) {
                stats->total_expansions++;
                stats->total_policy_cached++;
            }
        }
        pthread_mutex_unlock(&leaf->lock);
        return next;
    }
}

static inline void perform_backprop(Node *leaf, double value, MCTSConfig config, MCTSStats *stats) {
    backpropagate(leaf, value, config.use_solver);
    if (stats) stats->total_iterations++;
}

// =============================================================================
// SEQUENTIAL STEP
// =============================================================================

// Perform a single MCTS iteration (Sequential)
static void mcts_step_sequential(Node *root, Arena *arena, MCTSConfig config, MCTSStats *stats, TranspositionTable *tt) {
    // 1. Selection
    Node *leaf = perform_selection(root, config);
    
    if (leaf->is_terminal) {
        solve_terminal_node(leaf, config, stats);
        return;
    }

    // 2. Evaluation
    CNNOutput out;
    float value = 0.0f;
    float *policy_ptr = NULL;
    
    if (config.cnn_weights) {
        GameState *s0 = &leaf->state;
        GameState *s1 = (leaf->parent) ? &leaf->parent->state : NULL;
        GameState *s2 = (leaf->parent && leaf->parent->parent) ? &leaf->parent->parent->state : NULL;
        cnn_forward_with_history(config.cnn_weights, s0, s1, s2, &out);
        
        // Scale value from [-1, 1] to [0, 1] for MCTS backprop
        value = (out.value + 1.0f) / 2.0f;
        policy_ptr = out.policy;
    } else {
        value = (float)simulate_rollout(leaf, config);
    }

    // 3. Expansion
    Node *next_leaf = perform_expansion(leaf, arena, tt, config, policy_ptr, stats);

    // 4. Backpropagation
    perform_backprop(next_leaf, value, config, stats);
}

// =============================================================================
// THREAD POOL HELPERS
// =============================================================================

// Initialize inference queue for batch CNN processing
static void mcts_init_queue(InferenceQueue *queue) {
    queue->count = 0;
    queue->shutdown = 0;
    pthread_mutex_init(&queue->lock, NULL);
    pthread_cond_init(&queue->cond_batch_ready, NULL);
}

// Cleanup inference queue
static void mcts_destroy_queue(InferenceQueue *queue) {
    pthread_mutex_destroy(&queue->lock);
    pthread_cond_destroy(&queue->cond_batch_ready);
}

// Spawn worker threads for parallel MCTS iterations
static MCTSStats* mcts_spawn_workers(
    pthread_t *workers, WorkerArgs *args, int num_threads,
    Node *root, Arena *arena, MCTSConfig config, InferenceQueue *queue, TranspositionTable *tt
) {
    MCTSStats *worker_stats = calloc(num_threads, sizeof(MCTSStats));
    if (!worker_stats) return NULL;
    
    for (int i = 0; i < num_threads; i++) {
        args[i].root = root;
        args[i].arena = arena;
        args[i].config = config;
        args[i].queue = queue;
        args[i].tt = tt;
        args[i].thread_id = i;
        args[i].local_stats = &worker_stats[i];
        pthread_create(&workers[i], NULL, mcts_worker, &args[i]);
    }
    return worker_stats;
}

// Signal shutdown and join all worker threads
static void mcts_shutdown_workers(
    pthread_t *workers, int num_threads,
    InferenceQueue *queue,
    MCTSStats *worker_stats,
    long *out_iterations, long *out_expansions, long *out_children
) {
    queue->shutdown = 1;
    pthread_cond_broadcast(&queue->cond_batch_ready);
    
    // Wake up any waiting workers
    pthread_mutex_lock(&queue->lock);
    for (int i = 0; i < queue->count; i++) {
        pthread_cond_signal(&queue->requests[i]->cond);
    }
    pthread_mutex_unlock(&queue->lock);
    
    // Join and aggregate stats
    for (int i = 0; i < num_threads; i++) {
        pthread_join(workers[i], NULL);
        if (out_iterations) *out_iterations += worker_stats[i].total_iterations;
        if (out_expansions) *out_expansions += worker_stats[i].total_expansions;
        if (out_children)   *out_children   += worker_stats[i].total_policy_cached;
    }
}

// Added helper to merge worker stats into main stats safely
static void mcts_merge_worker_stats(MCTSStats *main_stats, MCTSStats *worker_stats, int num_threads) {
    if (!main_stats || !worker_stats) return;
    for (int i = 0; i < num_threads; i++) {
        main_stats->total_nodes += worker_stats[i].total_nodes;
        main_stats->total_depth += worker_stats[i].total_depth;
        main_stats->total_expansions += worker_stats[i].total_expansions;
        main_stats->total_policy_cached += worker_stats[i].total_policy_cached;
        main_stats->total_children_expanded += worker_stats[i].total_children_expanded;
        main_stats->nodes_with_children += worker_stats[i].nodes_with_children;
        main_stats->tt_hits += worker_stats[i].tt_hits;
        main_stats->tt_misses += worker_stats[i].tt_misses;
        if (worker_stats[i].peak_memory_bytes > main_stats->peak_memory_bytes) {
            main_stats->peak_memory_bytes = worker_stats[i].peak_memory_bytes;
        }
    }
}

// Process a batch of CNN inference requests
static void mcts_process_cnn_batch(InferenceQueue *queue, const CNNWeights *weights) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_nsec += 1000000; // 1ms timeout
    if (ts.tv_nsec >= 1000000000) {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1000000000;
    }
    
    pthread_mutex_lock(&queue->lock);
    
    // Wait for requests or timeout
    while (queue->count < 1 && !queue->shutdown) {
        pthread_cond_timedwait(&queue->cond_batch_ready, &queue->lock, &ts);
        if (queue->count > 0) break;
        
        struct timespec now;
        clock_gettime(CLOCK_REALTIME, &now);
        if (now.tv_sec > ts.tv_sec || (now.tv_sec == ts.tv_sec && now.tv_nsec >= ts.tv_nsec)) break;
    }
    
    if (queue->shutdown || queue->count == 0) {
        pthread_mutex_unlock(&queue->lock);
        return;
    }
    
    int current_batch = queue->count;
    
    // Batch arrays (stack allocated, limited by MCTS_BATCH_SIZE)
    CNNOutput outputs[MCTS_BATCH_SIZE];
    const GameState *states[MCTS_BATCH_SIZE];
    const GameState *hist1s[MCTS_BATCH_SIZE];
    const GameState *hist2s[MCTS_BATCH_SIZE];
    
    for (int i = 0; i < current_batch; i++) {
        Node *n = queue->requests[i]->node;
        states[i] = &n->state;
        hist1s[i] = (n->parent) ? &n->parent->state : NULL;
        hist2s[i] = (n->parent && n->parent->parent) ? &n->parent->parent->state : NULL;
    }
    
    cnn_forward_batch(weights, states, hist1s, hist2s, outputs, current_batch);
    
    // Distribute results to waiting threads
    for (int i = 0; i < current_batch; i++) {
        InferenceRequest *req = queue->requests[i];
        memcpy(req->policy_out, outputs[i].policy, CNN_POLICY_SIZE * sizeof(float));
        *req->value_out = (outputs[i].value + 1.0f) / 2.0f;
        req->ready = 1;
        pthread_cond_signal(&req->cond);
    }
    
    queue->count = 0;
    pthread_mutex_unlock(&queue->lock);
}

// Select the most visited child node (Robust Child selection)
static Node* mcts_select_best_child(Node *root) {
    Node *best = NULL;
    int max_visits = -1;
    
    for (int i = 0; i < root->num_children; i++) {
        int v = atomic_load(&root->children[i]->visits);
        if (v > max_visits) {
            max_visits = v;
            best = root->children[i];
        }
    }
    return best;
}

// Helper to recursively count expansions and children (if stats missing)
static void count_tree_expansions(Node *node, long *total_expansions, long *total_children) {
    if (!node) return;
    if (node->num_children > 0) {
        (*total_expansions)++;
        (*total_children) += node->num_children;
        for (int i = 0; i < node->num_children; i++) {
            count_tree_expansions(node->children[i], total_expansions, total_children);
        }
    }
}

// Update MCTSStats with results from this search
// Update MCTSStats with results from this search
static void mcts_update_stats(
    MCTSStats *stats, Node *root, 
    long iterations, long expansions, long children,
    double elapsed_time, size_t memory_used, int depth
) {
    if (!stats) return;
    
    stats->total_iterations += iterations;
    stats->total_nodes += get_tree_node_count(root);
    stats->current_move_iterations = iterations;
    stats->total_moves++;
    stats->total_depth += depth;
    stats->total_time += elapsed_time;
    stats->total_memory += memory_used;
    stats->total_expansions += expansions;
    stats->total_policy_cached += children;
    
    // Tree statistics: compute from tree structure if not provided by workers
    // OR if they look suspicious (expansions=0 but tree is large)
    if ((children == 0 && expansions == 0 && root) || (expansions == 0 && get_tree_node_count(root) > 100)) {
        long nodes_with_kids = 0;
        long total_kids = 0;
        count_tree_expansions(root, &nodes_with_kids, &total_kids);
        
        // DEBUG PRINT
        // printf("DEBUG FALLBACK: NodesWithKids=%ld TotalKids=%ld TotalNodes=%d\n", nodes_with_kids, total_kids, get_tree_node_count(root));

        stats->total_children_expanded += total_kids;
        stats->nodes_with_children += nodes_with_kids;
    } else {
        stats->total_children_expanded += children;
        stats->nodes_with_children += expansions;
    }
}

// =============================================================================
// MAIN MCTS SEARCH FUNCTION
// =============================================================================

Move mcts_search(Node *root, Arena *arena, double time_limit_seconds, MCTSConfig config,
                 MCTSStats *stats, TranspositionTable *tt, Node **out_new_root) {
    DBG_NOT_NULL(root);
    DBG_NOT_NULL(arena);
    struct timespec start_ts, current_ts;
    clock_gettime(CLOCK_MONOTONIC, &start_ts);
    
    // 0. Setup Queue
    InferenceQueue queue;
    mcts_init_queue(&queue);
    
    // 1. Spawn Workers (if num_threads > 0)
    int n_workers = config.num_threads;
    pthread_t workers[n_workers > 0 ? n_workers : 1];
    WorkerArgs args[n_workers > 0 ? n_workers : 1];
    MCTSStats *worker_stats_arr = NULL;

    if (n_workers > 0) {
        worker_stats_arr = mcts_spawn_workers(workers, args, n_workers,
                                               root, arena, config, &queue, tt);
    }
    
    // 2. Main Loop
    if (config.verbose) {
        printf("MCTS Start: Root=%p\n", root);
    }
    
    while (1) {
        // Check termination conditions
        clock_gettime(CLOCK_MONOTONIC, &current_ts);
        double elapsed = (current_ts.tv_sec - start_ts.tv_sec) + 
                        (current_ts.tv_nsec - start_ts.tv_nsec) / 1e9;
        int visits = atomic_load(&root->visits);
        int time_exceeded = (time_limit_seconds > 0) && (elapsed >= time_limit_seconds);
        int nodes_exceeded = (config.max_nodes > 0) && (visits >= config.max_nodes);
        
        if (time_exceeded || nodes_exceeded) break;

        // Early Exit Check (every EARLY_EXIT_CHECK_INTERVAL nodes to save CPU)
        if (visits > EARLY_EXIT_MIN_VISITS && visits % EARLY_EXIT_CHECK_INTERVAL == 0) {
            if (should_exit_early(root, config.max_nodes)) {
                break;
            }
        }
        
        if (n_workers == 0) {
            mcts_step_sequential(root, arena, config, stats, tt);
            continue;
        }

        if (!config.cnn_weights) {
            nanosleep(&(struct timespec){0, 1000000}, NULL); // 1ms sleep check
            continue;
        }

        // Process CNN batch requests
        mcts_process_cnn_batch(&queue, config.cnn_weights);
        if (queue.shutdown) break;
    }
    
    // 3. Cleanup & Join
    long iter_this_move = 0;
    long expansions_this_move = 0;
    long children_this_move = 0;

    if (n_workers > 0) {
        mcts_shutdown_workers(workers, n_workers, &queue, worker_stats_arr,
                              &iter_this_move, &expansions_this_move, &children_this_move);
        mcts_merge_worker_stats(stats, worker_stats_arr, n_workers);
        free(worker_stats_arr);
    } else {
        iter_this_move = root->visits;
    }
    
    mcts_destroy_queue(&queue);

    clock_gettime(CLOCK_MONOTONIC, &current_ts);
    double elapsed_time = (current_ts.tv_sec - start_ts.tv_sec) + 
                          (current_ts.tv_nsec - start_ts.tv_nsec) / 1e9;
    int depth = get_tree_depth(root);
    size_t memory_used = arena->offset;
    
    mcts_update_stats(stats, root, iter_this_move, expansions_this_move, 
                      children_this_move, elapsed_time, memory_used, depth);
    if (stats) {
        if (memory_used > stats->peak_memory_bytes) 
            stats->peak_memory_bytes = memory_used;
    }

    if (config.verbose) {
        printf("[MCTS Async] Tree depth: %d. Time: %.3fs. Memory: %.1f KB\n", 
               depth, elapsed_time, memory_used / 1024.0);
    }

    // Select best move (Robust Child: most visited)
    Node *best_child = mcts_select_best_child(root);

    if (best_child == NULL) {
        Move empty = {0};
        return empty;
    }

    if (config.use_tree_reuse && out_new_root) {
        *out_new_root = best_child;
    }

    return best_child->move_from_parent;
}

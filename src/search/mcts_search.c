/**
 * mcts_search.c - MCTS Main Search Algorithm
 * 
 * Contains: mcts_search, mcts_step_sequential, check_early_exit
 * Worker threads are in mcts_worker.c
 */

#include "dama/search/mcts.h"
#include "dama/search/mcts_tree.h"
#include "dama/search/mcts_worker.h"
#include "dama/neural/cnn.h"
#include "dama/common/params.h"
#include "dama/engine/movegen.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <sched.h>

// Forward declaration for rollout
double simulate_rollout(Node *node, MCTSConfig config);

// External functions from mcts_utils.c
extern int get_tree_depth(Node *node);
extern int get_tree_node_count(Node *node);

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Helper to determine game result
// Returns: 1 = White Win, 2 = Black Win, 0 = Draw, -1 = Ongoing
static int get_game_result(const GameState *state) {
    if (state->moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) return 0; // Draw
    
    MoveList ml;
    generate_moves(state, &ml);
    
    if (ml.count == 0) {
        // Current player has no moves -> Loses
        return (state->current_player == WHITE) ? 2 : 1;
    }
    
    return -1; // Ongoing
}

// Early Exit Check: Returns 1 if the best move cannot be overtaken
static int check_early_exit(Node *root, int max_nodes) {
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

static inline Node* perform_expansion(Node *leaf, Arena *arena, MCTSConfig config, float *policy, MCTSStats *stats) {
    if (config.cnn_weights) {
        if (config.verbose) {
            printf("Expansion: Selected %p. Ch=%d Untried=%d Term=%d\n", leaf, leaf->num_children, leaf->untried_moves.count, leaf->is_terminal);
        }
        
        pthread_mutex_lock(&leaf->lock);
        if (leaf->num_children == 0 && !leaf->is_terminal) {
            if (config.verbose) printf("Expanding %p\n", leaf);
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

static inline void perform_backprop(Node *leaf, double value, MCTSConfig config, MCTSStats *stats) {
    backpropagate(leaf, value, config.use_solver);
    if (stats) stats->total_iterations++;
}

// =============================================================================
// SEQUENTIAL STEP
// =============================================================================

// Perform a single MCTS iteration (Sequential)
static void mcts_step_sequential(Node *root, Arena *arena, MCTSConfig config) {
    // 1. Selection
    Node *leaf = perform_selection(root, config);
    
    if (leaf->is_terminal) {
        solve_terminal_node(leaf, config, NULL);
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
    Node *next_leaf = perform_expansion(leaf, arena, config, policy_ptr, NULL);

    // 4. Backpropagation
    perform_backprop(next_leaf, value, config, NULL);
}

// =============================================================================
// MAIN MCTS SEARCH FUNCTION
// =============================================================================

Move mcts_search(Node *root, Arena *arena, double time_limit_seconds, MCTSConfig config,
                 MCTSStats *stats, Node **out_new_root) {
    struct timespec start_ts, current_ts;
    clock_gettime(CLOCK_MONOTONIC, &start_ts);
    
    // 0. Setup Queue
    InferenceQueue queue;
    queue.count = 0;
    queue.shutdown = 0;
    pthread_mutex_init(&queue.lock, NULL);
    pthread_cond_init(&queue.cond_batch_ready, NULL);
    
    // 1. Spawn Workers (if NUM_MCTS_THREADS > 0)
    pthread_t workers[NUM_MCTS_THREADS > 0 ? NUM_MCTS_THREADS : 1];
    WorkerArgs args[NUM_MCTS_THREADS > 0 ? NUM_MCTS_THREADS : 1];
    MCTSStats *worker_stats_arr = NULL;

    if (NUM_MCTS_THREADS > 0) {
        worker_stats_arr = calloc(NUM_MCTS_THREADS, sizeof(MCTSStats));
        for (int i=0; i<NUM_MCTS_THREADS; i++) {
            args[i].root = root;
            args[i].arena = arena;
            args[i].config = config;
            args[i].queue = &queue;
            args[i].thread_id = i;
            args[i].local_stats = &worker_stats_arr[i];
            pthread_create(&workers[i], NULL, mcts_worker, &args[i]);
        }
    }
    
    // 2. Main Loop
    struct timespec ts;
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
            if (check_early_exit(root, config.max_nodes)) {
                break;
            }
        }
        
        if (NUM_MCTS_THREADS == 0) {
            mcts_step_sequential(root, arena, config);
            continue;
        }

        if (!config.cnn_weights) {
            nanosleep(&(struct timespec){0, 1000000}, NULL); // 1ms sleep check
            continue;
        }

        // CNN Batch Processing
        pthread_mutex_lock(&queue.lock);
        
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 1000000; // 1ms
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec += 1;
            ts.tv_nsec -= 1000000000;
        }
        
        while (queue.count < 1 && !queue.shutdown) {
             pthread_cond_timedwait(&queue.cond_batch_ready, &queue.lock, &ts);
             if (queue.count > 0) break;
             
             struct timespec now;
             clock_gettime(CLOCK_REALTIME, &now);
             if (now.tv_sec > ts.tv_sec || (now.tv_sec == ts.tv_sec && now.tv_nsec >= ts.tv_nsec)) break;
        }
        
        if (queue.shutdown) {
            pthread_mutex_unlock(&queue.lock);
            break;
        }
        
        if (queue.count == 0) {
            pthread_mutex_unlock(&queue.lock);
            continue; 
        }

        // --- PROCESS BATCH ---
        int current_batch = queue.count;
        
        // Local arrays for batch processing (stack allocated)
        CNNOutput outputs[MCTS_BATCH_SIZE];
        const GameState *states[MCTS_BATCH_SIZE];
        const GameState *hist1s[MCTS_BATCH_SIZE];
        const GameState *hist2s[MCTS_BATCH_SIZE];
        
        for (int i=0; i<current_batch; i++) {
             Node *n = queue.requests[i]->node;
             states[i] = &n->state;
             hist1s[i] = (n->parent) ? &n->parent->state : NULL;
             hist2s[i] = (n->parent && n->parent->parent) ? &n->parent->parent->state : NULL;
        }
        
        cnn_forward_batch(config.cnn_weights, states, hist1s, hist2s, outputs, current_batch);
        
        for (int i=0; i<current_batch; i++) {
            InferenceRequest *req = queue.requests[i];
            memcpy(req->policy_out, outputs[i].policy, CNN_POLICY_SIZE * sizeof(float));
            *req->value_out = (outputs[i].value + 1.0f) / 2.0f;
            req->ready = 1;
            pthread_cond_signal(&req->cond);
        }
        
        queue.count = 0;
        pthread_mutex_unlock(&queue.lock);
    }
    
    // 3. Cleanup & Join
    long iter_this_move = 0;
    long expansions_this_move = 0;
    long children_this_move = 0;

    queue.shutdown = 1;
    pthread_cond_broadcast(&queue.cond_batch_ready);
    
    if (NUM_MCTS_THREADS > 0) {
        pthread_mutex_lock(&queue.lock);
        for(int i=0; i<queue.count; i++) {
            pthread_cond_signal(&queue.requests[i]->cond);
        }
        pthread_mutex_unlock(&queue.lock);

        for (int i=0; i<NUM_MCTS_THREADS; i++) {
            pthread_join(workers[i], NULL);
            iter_this_move += worker_stats_arr[i].total_iterations;
            expansions_this_move += worker_stats_arr[i].total_expansions;
            children_this_move += worker_stats_arr[i].total_policy_cached;
        }
        free(worker_stats_arr);
    } else {
        iter_this_move = root->visits;
    }
    
    pthread_mutex_destroy(&queue.lock);
    pthread_cond_destroy(&queue.cond_batch_ready);

    clock_gettime(CLOCK_MONOTONIC, &current_ts);
    double elapsed_time = (current_ts.tv_sec - start_ts.tv_sec) + 
                          (current_ts.tv_nsec - start_ts.tv_nsec) / 1e9;
    int depth = get_tree_depth(root);
    size_t memory_used = arena->offset;
    
    if (stats) {
        stats->total_iterations += iter_this_move;
        stats->total_nodes += get_tree_node_count(root);
        stats->current_move_iterations = iter_this_move;
        
        stats->total_moves++;
        stats->total_depth += depth;
        stats->total_time += elapsed_time;
        stats->total_memory += memory_used;
        stats->total_expansions += expansions_this_move;
        stats->total_policy_cached += children_this_move;
    }

    if (config.verbose) {
        printf("[MCTS Async] Tree depth: %d. Time: %.3fs. Memory: %.1f KB\n", 
               depth, elapsed_time, memory_used / 1024.0);
    }

    // Select best move (Robust Child: most visited)
    Node *best_child = NULL;
    int max_visits = -1;

    for (int i = 0; i < root->num_children; i++) {
        int v = atomic_load(&root->children[i]->visits);
        if (v > max_visits) {
            max_visits = v;
            best_child = root->children[i];
        }
    }

    if (best_child == NULL) {
        Move empty = {0};
        return empty;
    }

    if (config.use_tree_reuse && out_new_root) {
        *out_new_root = best_child;
    }

    return best_child->move_from_parent;
}

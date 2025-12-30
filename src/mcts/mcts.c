/**
 * mcts.c - Monte Carlo Tree Search Main API
 * 
 * Contains: mcts_create_root, mcts_search, policy extraction
 */

#include "mcts.h"
#include "mcts_tree.h"
#include "../nn/cnn.h"
#include "../params.h"
#include "../core/movegen.h"
#include <time.h>
#include <stdio.h>

// Forward declaration for rollout
double simulate_rollout(Node *node, MCTSConfig config);

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

#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>

Node* mcts_create_root(GameState state, Arena *arena, MCTSConfig config) {
    Move no_move = {0};
    return create_node(NULL, no_move, state, arena, config);
}

int get_tree_depth(Node *node) {
    if (node->num_children == 0) return 0;
    
    int max_depth = 0;
    for (int i = 0; i < node->num_children; i++) {
        int child_depth = get_tree_depth(node->children[i]);
        if (child_depth > max_depth) max_depth = child_depth;
    }
    return max_depth + 1;
}

int get_tree_node_count(Node *node) {
    if (!node) return 0;
    int count = 1; // This node
    for (int i = 0; i < node->num_children; i++) {
        count += get_tree_node_count(node->children[i]);
    }
    return count;
}

// =============================================================================
// ASYNC BATCHING INFRASTRUCTURE
// =============================================================================

typedef struct {
    Node *node;
    float *policy_out;
    float *value_out;
    int ready;
    pthread_cond_t cond; // Condition variable for this specific request
} InferenceRequest;

typedef struct {
    GameState states[MCTS_BATCH_SIZE];
    InferenceRequest *requests[MCTS_BATCH_SIZE];
    int count;
    
    pthread_mutex_t lock;
    pthread_cond_t cond_batch_ready; // Signal Master that batch is ready/timer
    
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

// Worker Thread Function
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
            double result = 0.0;
            // Calculate result (simple game logic check)
            // Ideally should be cached or re-calculated
            // For now, assuming terminal nodes have score updated or handle in backprop?
            // Actually, terminal nodes don't need NN eval.
            // We need to simulate result from terminal state.
             int res = get_game_result(&leaf->state); // Need this function exposed or use check
             if (res == 1) result = (leaf->state.current_player == WHITE) ? 1.0 : 0.0; // Win for White
             else if (res == 2) result = (leaf->state.current_player == BLACK) ? 1.0 : 0.0; // Win for Black
             else result = config.draw_score;
             
             backpropagate(leaf, result, config.use_solver);
             if (args->local_stats) args->local_stats->total_iterations++;
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
            
            // Wait if queue full (backpressure)
            // For simplicity, just spin/wait or expand batch size.
            // With fixed threads and batch >= threads, queue never effectively "full" blocking forever
            // But if Master is slow?
            while (queue->count >= MCTS_BATCH_SIZE && !queue->shutdown) {
                // If full, signal master explicitly and wait?
                pthread_cond_signal(&queue->cond_batch_ready);
                pthread_mutex_unlock(&queue->lock);
                // Yield to let master run
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
            
            // Signal if full
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
            // Default policy (uniform for legal moves)
            // Expansion will fill legal moves.
        }

        // 3. Expansion
        // Lock is handled inside expand_node for the specific node
        // But we need to supply the policy.
        if (config.cnn_weights) {
             if (config.verbose) {
                 printf("Worker: Selected %p. Ch=%d Untried=%d Term=%d\n", leaf, leaf->num_children, leaf->untried_moves.count, leaf->is_terminal);
             }
             pthread_mutex_lock(&leaf->lock);
             if (leaf->num_children == 0 && !leaf->is_terminal) {
                  // Manual Expansion for Async:
                  if (config.verbose) printf("Expanding %p\n", leaf);
                  MoveList legal_moves;
                  generate_moves(&leaf->state, &legal_moves);
                  
                  if (legal_moves.count > 0) {
                      leaf->children = arena_alloc(args->arena, legal_moves.count * sizeof(Node*));
                      
                      // Filter policy for legal moves and normalize
                      float sum = 0.0f;
                      float *filtered_policy = arena_alloc(args->arena, legal_moves.count * sizeof(float));
                      
                      for (int i=0; i<legal_moves.count; i++) {
                           int idx = cnn_move_to_index(&legal_moves.moves[i], leaf->state.current_player);
                           float p = (idx >= 0) ? policy[idx] : 0.0f;
                           filtered_policy[i] = p;
                           sum += p;
                      }
                      
                      // Normalize and Spawn Children
                      for (int i=0; i<legal_moves.count; i++) {
                           if (sum > 1e-6) filtered_policy[i] /= sum;
                           else filtered_policy[i] = 1.0f / legal_moves.count;
                            GameState child_state = leaf->state;
                            apply_move(&child_state, &legal_moves.moves[i]);
                            
                            Node *child = create_node(leaf, legal_moves.moves[i], child_state, args->arena, config);
                            
                           child->prior = filtered_policy[i];
                           leaf->children[i] = child;
                      }
                      
                      // CRITICAL FIX: Mark node as fully expanded
                      leaf->untried_moves.count = 0;
                      
                      // Publish children safely
                      atomic_thread_fence(memory_order_release);
                      leaf->num_children = legal_moves.count;
                      
                      // Stats: CNN expanded all children at once
                      if (args->local_stats) {
                          args->local_stats->total_expansions++;
                          args->local_stats->total_policy_cached += legal_moves.count;
                      }

                  } else {
                      leaf->is_terminal = 1;
                  }
             } else {
                 // printf("Skipping Exp %p (Ch: %d)\n", leaf, leaf->num_children);
             }
             pthread_mutex_unlock(&leaf->lock);
        } else {
             // Vanilla Expansion
             pthread_mutex_lock(&leaf->lock);
             Node *next = leaf;
             if (!leaf->is_terminal) {
                 next = expand_node(leaf, args->arena, NULL, config);
                 // Stats: Vanilla expanded one child
                 if (args->local_stats) {
                     args->local_stats->total_expansions++;
                     args->local_stats->total_policy_cached++;  // 1 node
                 }
             }
             pthread_mutex_unlock(&leaf->lock);
             leaf = next;
        }
        
        // 4. Backpropagation
        backpropagate(leaf, value, config.use_solver);
        
        if (args->local_stats) args->local_stats->total_iterations++;
    }
    return NULL;
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

// Perform a single MCTS iteration (Sequential)
static void mcts_step_sequential(Node *root, Arena *arena, MCTSConfig config) {
    // 1. Selection
    Node *leaf = select_promising_node(root, config);
    
    if (leaf->is_terminal) {
        int res = get_game_result(&leaf->state);
        double result = 0.5;
        if (res == 1) result = (leaf->state.current_player == WHITE) ? 1.0 : 0.0;
        else if (res == 2) result = (leaf->state.current_player == BLACK) ? 1.0 : 0.0;
        else result = config.draw_score;
        backpropagate(leaf, result, config.use_solver);
        return;
    }

    // 2. Evaluation
    CNNOutput out;
    float value = 0.0f;
    if (config.cnn_weights) {
        GameState *s0 = &leaf->state;
        GameState *s1 = (leaf->parent) ? &leaf->parent->state : NULL;
        GameState *s2 = (leaf->parent && leaf->parent->parent) ? &leaf->parent->parent->state : NULL;
        cnn_forward_with_history(config.cnn_weights, s0, s1, s2, &out);
        
        // VALUE SCALE FIX: MCTS backprop (1-r logic) expects [0, 1] range.
        // CNN outputs [-1, 1], so we map it: -1.0 -> 0.0 (Loss), 1.0 -> 1.0 (Win)
        value = (out.value + 1.0f) / 2.0f;
    } else {
        value = (float)simulate_rollout(leaf, config);
    }

    // 3. Expansion
    if (config.cnn_weights) {
        MoveList legal_moves;
        generate_moves(&leaf->state, &legal_moves);
        if (legal_moves.count > 0) {
            leaf->children = arena_alloc(arena, legal_moves.count * sizeof(Node*));
            float sum = 0.0f;
            float filtered_policy[CNN_POLICY_SIZE];
            for (int i=0; i<legal_moves.count; i++) {
                int idx = cnn_move_to_index(&legal_moves.moves[i], leaf->state.current_player);
                float p = (idx >= 0) ? out.policy[idx] : 0.0f;
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
            leaf->num_children = legal_moves.count;
        } else {
            leaf->is_terminal = 1;
        }
    } else {
        Node *next = expand_node(leaf, arena, NULL, config);
        leaf = next;
    }

    // 4. Backpropagation
    backpropagate(leaf, value, config.use_solver);
}

// MAIN MCTS SEARCH FUNCTION (MASTER THREAD)
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
    
    // 2. Master Loop (Batch Processor)
    // Only runs if using CNN. If Vanilla, Master just sleeps/waits or (better) joins?
    // Vanilla workers don't use the queue wait loop.
    // But we need to enforce timeout.
    
    struct timespec ts;
    if (config.verbose) {
        printf("MCTS Start: Root=%p\n", root);
    }
    
    // 3. Main Loop
    while (1) {
        // Check termination conditions
        clock_gettime(CLOCK_MONOTONIC, &current_ts);
        double elapsed = (current_ts.tv_sec - start_ts.tv_sec) + 
                        (current_ts.tv_nsec - start_ts.tv_nsec) / 1e9;
        int visits = atomic_load(&root->visits);
        // time_limit_seconds <= 0 means unlimited time (use node limit only)
        int time_exceeded = (time_limit_seconds > 0) && (elapsed >= time_limit_seconds);
        int nodes_exceeded = (config.max_nodes > 0) && (visits >= config.max_nodes);
        
        if (time_exceeded || nodes_exceeded) break;

        // Early Exit Check (every 10 nodes to save CPU)
        if (visits > 40 && visits % 10 == 0) {
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
        
        // Wait for batch or timeout (e.g. 1ms max latency)
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 1000000; // 1ms
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec += 1;
            ts.tv_nsec -= 1000000000;
        }
        
        // Wait for batch or timeout (1ms)
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
        
        // Local buffers
        static CNNOutput outputs[MCTS_BATCH_SIZE];
        static const GameState *states[MCTS_BATCH_SIZE];
        static const GameState *hist1s[MCTS_BATCH_SIZE];
        static const GameState *hist2s[MCTS_BATCH_SIZE];
        
        // 1. Extract states and history for batch processing
        for (int i=0; i<current_batch; i++) {
             Node *n = queue.requests[i]->node;
             states[i] = &n->state;
             hist1s[i] = (n->parent) ? &n->parent->state : NULL;
             hist2s[i] = (n->parent && n->parent->parent) ? &n->parent->parent->state : NULL;
        }
        
        // 2. Run CNN Inference (TRUE BATCH with sgemm optimization)
        cnn_forward_batch(config.cnn_weights, states, hist1s, hist2s, outputs, current_batch);
        
        // 2. Distribute Results
        for (int i=0; i<current_batch; i++) {
            InferenceRequest *req = queue.requests[i];
            memcpy(req->policy_out, outputs[i].policy, CNN_POLICY_SIZE * sizeof(float));
            
            // VALUE SCALE FIX: MCTS backprop (1-r logic) expects [0, 1] range.
            // CNN outputs [-1, 1], so we map it: -1.0 -> 0.0 (Loss), 1.0 -> 1.0 (Win)
            *req->value_out = (outputs[i].value + 1.0f) / 2.0f;
            
            req->ready = 1;
            pthread_cond_signal(&req->cond);
        }
        
        queue.count = 0; // Reset
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
        stats->total_nodes += get_tree_node_count(root); // Count tree nodes
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

// =============================================================================
// POLICY EXTRACTION (AlphaZero)
// =============================================================================

void mcts_get_policy(Node *root, float *policy, float temperature, const GameState *state) {
    memset(policy, 0, CNN_POLICY_SIZE * sizeof(float));
    
    if (!root || root->num_children == 0) return;
    
    // 1. Collect visits
    double total_visits = 0;
    for (int i = 0; i < root->num_children; i++) {
        total_visits += root->children[i]->visits;
    }
    
    if (total_visits < 1) return;
    
    // 2. Argmax Case (Temp -> 0)
    if (temperature < 1e-3) {
        double max_visits = -1.0;
        Node *best_child = NULL;
        
        for (int i = 0; i < root->num_children; i++) {
            Node *child = root->children[i];
            if (child->visits > max_visits) {
                max_visits = child->visits;
                best_child = child;
            }
        }
        
        if (best_child) {
            int idx = cnn_move_to_index(&best_child->move_from_parent, state->current_player);
            if (idx >= 0 && idx < CNN_POLICY_SIZE) {
                policy[idx] = 1.0f;
            }
        }
        return;
    }
    
    // 3. Proportional Case (with Temperature)
    // pi(a) = (N(a)^(1/tau)) / Sum(...)
    double sum_pow = 0.0;
    double exponent = 1.0 / temperature;
    
    // First pass: sum exponentiated visits
    for (int i = 0; i < root->num_children; i++) {
        sum_pow += pow(root->children[i]->visits, exponent);
    }
    
    // Second pass: normalize
    for (int i = 0; i < root->num_children; i++) {
        Node *child = root->children[i];
        int idx = cnn_move_to_index(&child->move_from_parent, state->current_player);
        if (idx >= 0 && idx < CNN_POLICY_SIZE) {
            double p = pow(child->visits, exponent);
            policy[idx] = (float)(p / sum_pow);
        }
    }
}
// =============================================================================
// DIAGNOSTICS & DEBUG
// =============================================================================

double mcts_get_avg_root_ucb(Node *root, MCTSConfig config) {
    if (!root || root->num_children == 0) return 0.0;
    
    double total_ucb = 0.0;
    int count = 0;
    
    for (int i = 0; i < root->num_children; i++) {
        double val = calculate_ucb1_score(root->children[i], config);
        if (val < 1e8) { 
            total_ucb += val;
            count++;
        }
    }
    
    return (count > 0) ? (total_ucb / count) : 0.0;
}

// Compare function for qsort
static int compare_nodes_visits_desc(const void *a, const void *b) {
    Node *nodeA = *(Node **)a;
    Node *nodeB = *(Node **)b;
    return nodeB->visits - nodeA->visits;
}

void print_mcts_stats_sorted(Node *root) {
    if (!root || root->num_children == 0) {
        printf("No children statistics available.\n");
        return;
    }

    Node **sorted_children = malloc(root->num_children * sizeof(Node*));
    if (!sorted_children) return;

    for (int i=0; i<root->num_children; i++) {
        sorted_children[i] = root->children[i];
    }

    qsort(sorted_children, root->num_children, sizeof(Node*), compare_nodes_visits_desc);

    // Dummy config for UCB calculation (conservative defaults)
    MCTSConfig dummy_cfg = {0};
    dummy_cfg.ucb1_c = UCB1_C;
    dummy_cfg.puct_c = PUCT_C;
    dummy_cfg.use_puct = 1;

    printf("--- Root Children Stats (Sorted by Visits) ---\n");
    for (int i = 0; i < root->num_children; i++) {
        Node *child = sorted_children[i];
        printf("Move: ");
        print_move_description(child->move_from_parent);
        
        double win_rate = (child->visits > 0) ? (child->score / child->visits) : 0.0;
        int depth = get_tree_depth(child);
        double ucb = calculate_ucb1_score(child, dummy_cfg);
        
        printf(" | Visits: %d | Depth: %d | WinRate: %.1f%% | UCB: %.3f\n", 
               child->visits, depth, win_rate * 100.0, ucb);
    }
    printf("----------------------------------------------\n");

    free(sorted_children);
}

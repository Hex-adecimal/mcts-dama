#include "mcts.h"
#include "debug.h"
#include "params.h"

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

// =============================================================================
// MEMORY MANAGEMENT (Arena Allocator)
// =============================================================================
// arena_init, arena_alloc, arena_reset, arena_free

void arena_init(Arena *a, size_t total_size) {
    a->buffer = malloc(total_size);
    if (!a->buffer) { 
        fprintf(stderr, "FATAL: Malloc failed for Arena of size %zu\n", total_size); 
        exit(1); 
    }
    a->size = total_size;
    a->offset = 0; 
}

void* arena_alloc(Arena *a, size_t bytes) {
    // Align pointer to 8 bytes to avoid alignment faults on some architectures
    uintptr_t current_ptr = (uintptr_t)(a->buffer + a->offset);
    uintptr_t padding = (8 - (current_ptr % 8)) % 8; 
    
    if (a->offset + padding + bytes > a->size) {
        fprintf(stderr, "FATAL: Arena Out of Memory! (Size: %zu)\n", a->size);
        exit(1);
    }
    
    void *ptr = a->buffer + a->offset + padding;
    a->offset += padding + bytes;
    return ptr;
}

void arena_reset(Arena *a) { 
    a->offset = 0; 
}

void arena_free(Arena *a) { 
    free(a->buffer); 
}

// ================================================================================================
//  HEURISTICS
// ================================================================================================

/**
 * Evaluates a move based on heuristics (Promotion, Safety, Center Control).
 * Higher score = better move.
 */
static double evaluate_move_heuristic(const GameState *state, const Move *move, MCTSConfig config) {
    double score = 0.0;
    int us = state->current_player;
    
    int target_idx = (move->length == 0) ? 1 : move->length;
    int from = move->path[0];
    int to   = move->path[target_idx];
    
    int row = to / 8;
    int col = to % 8;
    int from_row = from / 8;

    // 1. Capture Bonus
    if (move->length > 0) {
        score += config.weights.w_capture * move->length; 
    }

    // 2. Promotion Bonus
    if (!move->is_lady_move) {
        if (row == 0 || row == 7) { // Reaching end
            score += config.weights.w_promotion;
        }
        
        // Advancement Bonus
        int dist = (us == WHITE) ? (7 - row) : row;
        score += (7 - dist) * config.weights.w_advance;
    }

    // 3. Safety Bonus (Edges) - ONLY for regular pieces, not queens
    // Queens should stay active in the center, not hide in corners
    if (!move->is_lady_move && (col == 0 || col == 7)) {
        score += config.weights.w_edge;
    }

    // 4. Center Control
    if ((row == 3 || row == 4) && (col >= 2 && col <= 5)) {
        score += config.weights.w_center; 
    }

    // 5. Base Protection (Penalty)
    if (!move->is_lady_move) {
         if ((us == WHITE && from_row == 0) || (us == BLACK && from_row == 7)) {
             score -= config.weights.w_base;
         }
    }
    
    // 6. Queen Activity Bonus - encourage queens to move
    if (move->is_lady_move) {
        score += config.weights.w_lady_activity;
    }
    
    return score;
}

// ================================================================================================
//  NODE MANAGEMENT
// ================================================================================================

/**
 * Creates a new MCTS node.
 * @param parent Pointer to the parent node (NULL for root).
 * @param move The move that led to this state.
 * @param state The game state resulting from the move.
 * @param arena Pointer to the arena for allocation.
 * @return Pointer to the new Node.
 */
static Node* create_node(Node *parent, Move move, GameState state, Arena *arena, MCTSConfig config) {
    Node *node = (Node*)arena_alloc(arena, sizeof(Node));
    
    node->state = state;
    node->move_from_parent = move;
    node->parent = parent;
    
    // If it's White's turn in the new state, it means Black just moved.
    node->player_who_just_moved = (state.current_player == WHITE) ? BLACK : WHITE;

    node->children = NULL;
    node->num_children = 0;
    node->visits = 0;
    node->score = 0.0;
    node->sum_sq_score = 0.0; // Init variance sum

    // Generate all possible moves from this state (untried moves)
    generate_moves(&node->state, &node->untried_moves);
    node->is_terminal = (node->untried_moves.count == 0 && node->num_children == 0) ? 1 : 0; 
    
    // SOLVER INIT
    node->status = SOLVED_NONE;
    if (node->is_terminal) {
        // If no moves, I lost. (Assuming no stale-mate draw logic here for now)
        node->status = SOLVED_LOSS; 
        
        if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
             node->status = SOLVED_DRAW;
        }
    }
    
    // HEURISTIC INIT (Progressive Bias)
    if (parent) {
        node->heuristic_score = evaluate_move_heuristic(&parent->state, &move, config);
        
        // Dynamic Threat Check (only if weight > 0) to avoid overhead
        if (config.weights.w_threat > 0.0) {
            // Determine destination square
            int dest = (move.length == 0) ? move.path[1] : move.path[move.length];
            
            // Check if this piece is immediately threatened in the new state
            if (is_square_threatened(&node->state, dest)) {
                node->heuristic_score -= config.weights.w_threat;
            }
        }
    } else {
        node->heuristic_score = 0.0;
    }

    return node;
}

/**
 * Compares two moves for equality.
 * @param m1 First move.
 * @param m2 Second move.
 * @return 1 if moves are equal, 0 otherwise.
 */
// Helper to compare moves
static int moves_equal(const Move *m1, const Move *m2) {
    if (m1->length != m2->length) return 0;
    
    // For simple moves (len=0), path has 2 elements (start, end).
    // For captures (len>0), path has len+1 elements.
    // If len=0, we must check index 0 AND index 1.
    int limit = (m1->length == 0) ? 1 : m1->length;

    for (int i = 0; i <= limit; i++) {
        if (m1->path[i] != m2->path[i]) return 0;
    }
    return 1;
}

/**
 * Finds child node matching the given move.
 * Used for tree reuse to locate opponent's move in our tree.
 * @param parent Parent node to search in.
 * @param move Move to find.
 * @return Child node if found, NULL otherwise.
 */
Node* find_child_by_move(Node *parent, const Move *move) {
    if (!parent || !move) return NULL;
    
    for (int i = 0; i < parent->num_children; i++) {
        if (moves_equal(&parent->children[i]->move_from_parent, move)) {
            return parent->children[i];
        }
    }
    return NULL;
}

// ================================================================================================
//  TRANSPOSITION TABLE
// ================================================================================================

// Simple Hash Table with Linear Probing (or simpler Direct Mapping for speed)
// For MCTS, we can use a large enough table and just overwrite or collision chain.
// Let's use Open Addressing with Linear Probing for simplicity and cache locality.

static TranspositionTable* tt_create(size_t size) {
    TranspositionTable *tt = malloc(sizeof(TranspositionTable));
    tt->size = size;
    tt->mask = size - 1;
    tt->count = 0;
    tt->collisions = 0;
    tt->buckets = calloc(size, sizeof(Node*)); // Init to NULL
    return tt;
}

static void tt_free(TranspositionTable *tt) {
    if (tt) {
        free(tt->buckets);
        free(tt);
    }
}

// Helper to compare states fully
int states_equal(const GameState *s1, const GameState *s2) {
    if (s1->white_pieces != s2->white_pieces) return 0;
    if (s1->black_pieces != s2->black_pieces) return 0;
    if (s1->white_ladies != s2->white_ladies) return 0;
    if (s1->black_ladies != s2->black_ladies) return 0;
    if (s1->current_player != s2->current_player) return 0;
    // moves_without_captures affects Draw rule, so it matters for state identity
    if (s1->moves_without_captures != s2->moves_without_captures) return 0;
    return 1;
}

static Node* tt_lookup(TranspositionTable *tt, const GameState *state) {
    if (!tt) return NULL;
    size_t idx = state->hash & tt->mask;
    
    // Simple direct lookup for now (soft collision handling: just overwrite/ignore)
    // For a real robust implementation we would do probing.
    
    Node *node = tt->buckets[idx];
    if (node && node->state.hash == state->hash) {
        // Double check full state equality to be safe (hash collisions are rare but possible)
        if (states_equal(&node->state, state)) {
            return node;
        }
    }
    return NULL; 
}

static void tt_insert(TranspositionTable *tt, Node *node) {
    if (!tt) return;
    size_t idx = node->state.hash & tt->mask;
    
    // Always overwrite (Replacement scheme: Always Replace)
    if (tt->buckets[idx] != NULL) tt->collisions++;
    tt->buckets[idx] = node;
    tt->count++;
}

/**
 * Expands a leaf node by adding one child for an untried move.
 * @param node The node to expand.
 * @param arena Pointer to the arena.
 * @return Pointer to the newly created child node.
 */
static Node* expand_node(Node *node, Arena *arena, TranspositionTable *tt, MCTSConfig config) {
    if (node->untried_moves.count == 0) return node; // Should not happen if checked before

    // Pop the last move from untried_moves (efficient)
    int idx = node->untried_moves.count - 1;
    Move move_to_try = node->untried_moves.moves[idx];
    node->untried_moves.count--;

    // Apply the move to get the new state
    GameState next_state = node->state;
    apply_move(&next_state, &move_to_try);

    // Create the child node
    Node *child = create_node(node, move_to_try, next_state, arena, config);

    // Handle Transposition Table
    if (tt) {
        Node *match = tt_lookup(tt, &next_state);
        if (match) {
            // Found in TT: Warm-start this node with stats from the transposition!
            child->visits = match->visits;
            child->score  = match->score;
            child->sum_sq_score = match->sum_sq_score;
            child->status = match->status; // Copy solved status
        }
        
        // Always update TT to point to the new active representative of this state
        // This ensures the TT entry points to the node currently in the tree,
        // so it receives updates during backpropagation.
        tt_insert(tt, child);
    }

    // Resize children array using Arena
    size_t new_size = (node->num_children + 1) * sizeof(Node*);
    Node **new_children = (Node**)arena_alloc(arena, new_size);

    if (node->num_children > 0) {
        memcpy(new_children, node->children, node->num_children * sizeof(Node*));
    }
    
    new_children[node->num_children] = child;
    node->children = new_children;
    node->num_children++;

    return child;
}

// ================================================================================================
//  SELECTION (UCB1)
// ================================================================================================

/**
 * Calculates the UCB1 value for a node.
 * UCB1 = WinRate + C * sqrt(ln(ParentVisits) / NodeVisits)
 */
static double calculate_ucb1(Node *child, MCTSConfig config) {
    if (child->visits == 0) {
        return config.use_fpu ? config.fpu_value : 1e9;
    }

    double win_rate = child->score / (double)child->visits;
    double exploration = UCB1_C * sqrt(log((double)child->parent->visits) / (double)child->visits);

    return win_rate + exploration;
}

/**
 * Calculates the UCB1-Tuned value for a node.
 * Uses variance estimate to tune the difference interval.
 * Formula: UCB = Mean + sqrt( (ln N / n) * min(1/4, V) )
 * Where V = Variance + sqrt( (2 ln N) / n )
 */
static double calculate_ucb1_tuned(Node *child, MCTSConfig config) {
    if (child->visits == 0) {
        return config.use_fpu ? config.fpu_value : 1e9;
    }

    double N = (double)child->parent->visits;
    double n = (double)child->visits;
    
    double mean = child->score / n;
    
    // Calculate variance of rewards
    // Var(X) = E[X^2] - (E[X])^2
    double avg_sq_score = child->sum_sq_score / n;
    double variance = avg_sq_score - (mean * mean);
    
    // Variance Upper Bound
    double v_upper = variance + sqrt(2.0 * log(N) / n);

    // Tuned exploration term
    // min(1/4, v_upper) because max variance of a bounded [0,1] var consists of 0.25
    double min_v = (v_upper < 0.25) ? v_upper : 0.25;
    double exploration = sqrt((log(N) / n) * min_v);

    return mean + exploration;
}

double calculate_ucb1_score(Node *child, MCTSConfig config) {
    double base_score;
    if (config.use_ucb1_tuned) {
        base_score = calculate_ucb1_tuned(child, config);
    } else {
        base_score = calculate_ucb1(child, config);
    }
    
    if (config.use_progressive_bias) {
        double bias = config.bias_constant * (child->heuristic_score / (double)(child->visits + 1));
        return base_score + bias;
    }
    
    return base_score;
}

/**
 * Selects the best child node to explore using UCB1 or UCB1-Tuned.
 * Descends the tree until a node with untried moves or a terminal node is reached.
 */
// Returns the best child node based on UCB1 or UCB1-Tuned
static Node* select_promising_node(Node *root, MCTSConfig config) {
    Node *current = root;
    while (!current->is_terminal && current->untried_moves.count == 0) {
        if (current->num_children == 0) break; // Should be covered by is_terminal logic

        // SOLVER LOGIC: Taking the win immediately
        if (config.use_solver && current->status == SOLVED_WIN) {
            // Find the child that gives the win (opponent loss)
            for (int i=0; i < current->num_children; i++) {
                if (current->children[i]->status == SOLVED_LOSS) {
                    current = current->children[i];
                    goto next_node; // Jump to next iteration to avoid UCB calculation
                }
            }
        }
        
        double best_score = -1e9;
        Node *best_node = NULL;
        
        for (int i = 0; i < current->num_children; i++) {
            Node *child = current->children[i];
            double ucb_value;
            
            // SOLVER PRUNING:
            if (config.use_solver) {
                 if (child->status == SOLVED_WIN) {
                     // This child leads to Opponent Win. Bad for us.
                     // Assign penalty unless all are bad.
                     ucb_value = -100000.0; 
                 } else if (child->status == SOLVED_LOSS) {
                     // This child leads to Opponent Loss. We WIN.
                     // Should be picked immediately above, but if we are here,
                     // it means we missed it? No, handled above.
                     ucb_value = 100000.0 + child->score; // Very High
                 } else {
                     // Normal UCB with potential bias
                     ucb_value = calculate_ucb1_score(child, config);
                 }
            } else {
                 ucb_value = calculate_ucb1_score(child, config);
            }
            
            if (ucb_value > best_score) {
                best_score = ucb_value;
                best_node = child;
            }
        }
        current = best_node;
        
        next_node:;
    }
    return current;
}

// ================================================================================================
//  SIMULATION (ROLLOUT)
// ================================================================================================

/**
 * Heuristic to pick a "smart" move during rollout instead of purely random.
 * Gives bonuses for promotion, safety, etc.
 * Optionally uses 1-ply lookahead to avoid dangerous moves.
 */
static Move pick_smart_move(const MoveList *list, const GameState *state, int use_lookahead, MCTSConfig config) {
    if (list->count == 1) return list->moves[0];

    int best_score = -100000;
    int best_idx = 0; 

    // Evaluate moves
    for (int i = 0; i < list->count; i++) {
        Move m = list->moves[i];
        int score = 0;
        
        // Prefer captures (already handled by rule, but good for robust fallback)
        if (m.length > 0) score += 1000 * m.length;
        
        // Use shared heuristic function
        score += (int)evaluate_move_heuristic(state, &m, config);

        // Danger Check (1-ply lookahead)
        // Only check for simple moves (captures are usually good)
        // AND only in endgame positions (< 12 total pieces) where tactics matter most
        if (use_lookahead && m.length == 0) {
            // Count total pieces on board
            int total_pieces = __builtin_popcountll(state->white_pieces | state->black_pieces | 
                                                     state->white_ladies | state->black_ladies);
            
            // Activate lookahead only in endgame (tactical phase)
            if (total_pieces < 12) {
                GameState future_state = *state;
                apply_move(&future_state, &m);
                
                MoveList enemy_moves;
                generate_moves(&future_state, &enemy_moves);
                
                // If opponent can capture after this move, it's dangerous
                if (enemy_moves.count > 0 && enemy_moves.moves[0].length > 0) {
                    score -= WEIGHT_DANGER;
                }
            }
        }

        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    return list->moves[best_idx];
}

/**
 * Simulates a random game from the given node to determine a winner.
 * @return Winner color (WHITE/BLACK) or -1 for Draw.
 */
static double simulate_rollout(Node *node, MCTSConfig config) {
    GameState temp_state = node->state;
    
    // Check if the node itself is terminal
    if (node->is_terminal) {
        int winner = (temp_state.current_player == WHITE) ? BLACK : WHITE;
        if (winner == node->player_who_just_moved) return WIN_SCORE;
        else return LOSS_SCORE;
    }

    int depth = 0;
    MoveList temp_moves;
    
    // Limit depth to avoid infinite loops
    while (depth < MAX_ROLLOUT_DEPTH) {
        generate_moves(&temp_state, &temp_moves);

        if (temp_moves.count == 0) {
            int winner = (temp_state.current_player == WHITE) ? BLACK : WHITE;
            if (winner == node->player_who_just_moved) {
                double score = WIN_SCORE;
                if (config.use_decaying_reward) {
                    score *= pow(config.decay_factor, depth);
                }
                return score;
            } else {
                return LOSS_SCORE;
            }
        }
        
        if (temp_state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
            return DRAW_SCORE; // Draw
        }

        Move chosen_move;

        // Epsilon-Greedy Strategy:
        // With probability epsilon, choose random move.
        // Otherwise, choose "smart" move.
        double r = (double)rand() / RAND_MAX;

        if (r < config.rollout_epsilon) {
            int random_idx = rand() % temp_moves.count;
            chosen_move = temp_moves.moves[random_idx];
        } else {
            chosen_move = pick_smart_move(&temp_moves, &temp_state, config.use_lookahead, config);
        }

        apply_move(&temp_state, &chosen_move);
        depth++;
    }
    return DRAW_SCORE; // Draw if depth limit reached
}

// ================================================================================================
//  BACKPROPAGATION
// ================================================================================================

/**
 * Propagates the simulation result back up the tree.
 * Updates visits and scores for all nodes in the path.
 */
// Helper to check if a node is fully expanded and solved
static void update_solver_status(Node *node) {
    if (node->status != SOLVED_NONE) return; // Already solved
    
    // Condition 1: Can we win immediately?
    // If ANY child is SOLVED_LOSS (for the opponent), implies SOLVED_WIN for us.
    for (int i = 0; i < node->num_children; i++) {
        if (node->children[i]->status == SOLVED_LOSS) {
            node->status = SOLVED_WIN;
            // Best move is to go here!
            return; 
        }
    }

    // Condition 2: Are we forced to lose?
    // Must be fully expanded (no untried moves) AND all children are SOLVED_WIN (for the opponent).
    if (node->untried_moves.count == 0) {
        int all_win = 1;
        for (int i = 0; i < node->num_children; i++) {
            if (node->children[i]->status != SOLVED_WIN) {
                all_win = 0;
                break;
            }
        }
        
        if (all_win && node->num_children > 0) { // If num_children=0, handled in init (Terminal)
            node->status = SOLVED_LOSS;
        }
    }
}

static void backpropagate(Node *node, double result, int use_solver) {
    Node *child = NULL; // Keep track of the node we came from
    
    while (node != NULL) {
        node->visits++;
        
        // Add Result (from this node's player perspective)
        node->score += result;
        node->sum_sq_score += (result * result);

        // SOLVER UPDATE
        if (use_solver) {
            if (child == NULL || child->status != SOLVED_NONE) {
                update_solver_status(node);
            }
        }

        // CRITICAL: Flip perspective for the parent node!
        // In a two-player game, if this node's player won (result=1),
        // that means the parent's player lost (result=0), and vice versa.
        result = 1.0 - result;

        child = node; // Move up
        node = node->parent;
    }
}

// ================================================================================================
//  PUBLIC API
// ================================================================================================

Node* mcts_create_root(GameState state, Arena *arena, MCTSConfig config) {
    Move no_move = {0};
    return create_node(NULL, no_move, state, arena, config);
}

/**
 * Calculates the maximum depth of the MCTS tree.
 * Uses recursive traversal to find the deepest leaf.
 */
int get_tree_depth(Node *node) {
    if (node->num_children == 0) return 0;
    
    int max_depth = 0;
    for (int i = 0; i < node->num_children; i++) {
        int child_depth = get_tree_depth(node->children[i]);
        if (child_depth > max_depth) max_depth = child_depth;
    }
    return max_depth + 1;
}

Move mcts_search(Node *root, Arena *arena, double time_limit_seconds, MCTSConfig config,
                 MCTSStats *stats, Node **out_new_root) {
    clock_t start = clock();
    int iterations = 0;

    // Initialize Transposition Table if enabled
    TranspositionTable *tt = NULL;
    if (config.use_tt) {
        // Size should be large power of 2. 
        // 1M entries * 8 bytes = 8MB. Cheap.
        tt = tt_create(1024 * 1024); 
    }

    // Main MCTS Loop
    do {
        // SAFETY CHECK: Stop if Arena is nearly full to prevent OOM crash
        if (arena->offset > arena->size * 0.95) {
            fprintf(stderr, "[WARNING] MCTS Memory Limit Hit: %.1f MB / %.1f MB used. Stopping search early.\n", 
                    (double)arena->offset / (1024.0 * 1024.0), (double)arena->size / (1024.0 * 1024.0));
            break; 
        }

        // 1. Selection
        Node *leaf = select_promising_node(root, config);
        
        // 2. Expansion
        if (!leaf->is_terminal) {
            leaf = expand_node(leaf, arena, tt, config); 
        }

        // 3. Simulation
        double result = simulate_rollout(leaf, config);
        
        // 4. Backpropagation
        backpropagate(leaf, result, config.use_solver);
        
        iterations++;
    } while ( (double)(clock() - start) / CLOCKS_PER_SEC < time_limit_seconds );

    int depth = get_tree_depth(root);
    double elapsed_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    size_t memory_used = arena->offset; // Bytes used in arena
    
    // Update statistics if provided
    if (stats) {
        stats->total_moves++;
        stats->total_iterations += iterations;
        stats->total_depth += depth;
        stats->total_time += elapsed_time;
        stats->total_memory += memory_used;
    }

    if (config.verbose) {
        printf("[MCTS] %d simulations. Tree depth: %d. Time: %.3fs (%.0f iter/s). Memory: %.1f KB\n", 
               iterations, depth, elapsed_time, iterations / elapsed_time, memory_used / 1024.0);
    }

    // Select best move (Robust Child: most visited)
    Node *best_child = NULL;
    int max_visits = -1;

    for (int i = 0; i < root->num_children; i++) {
        if (root->children[i]->visits > max_visits) {
            max_visits = root->children[i]->visits;
            best_child = root->children[i];
        }
    }

    if (best_child == NULL) {
        fprintf(stderr, "MCTS panic: No moves available!\n");
        Move empty = {0};
        return empty;
    }

    // If tree reuse enabled, return the chosen child as new root
    if (config.use_tree_reuse && out_new_root) {
        *out_new_root = best_child;
    }

    if (tt) tt_free(tt);

    return best_child->move_from_parent;
}
#include "mcts.h"
#include "params.h"

#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

// ================================================================================================
//  DEBUG HELPERS
// ================================================================================================

void print_move_description(Move m) {
    int target_idx = (m.length == 0) ? 1 : m.length;
    
    printf("%c%d -> %c%d", 
           (m.path[0]%8)+'A', (m.path[0]/8)+1,
           (m.path[target_idx]%8)+'A', (m.path[target_idx]/8)+1);
           
    if (m.length > 0) printf(" (CAPTURE)");
    printf("\n");
}

// ================================================================================================
//  MEMORY MANAGEMENT (ARENA)
// ================================================================================================

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
static Node* create_node(Node *parent, Move move, GameState state, Arena *arena) {
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

    // Generate all possible moves from this state (untried moves)
    generate_moves(&node->state, &node->untried_moves);
    node->is_terminal = (node->untried_moves.count == 0);

    return node;
}

/**
 * Expands a leaf node by adding one child for an untried move.
 * @param node The node to expand.
 * @param arena Pointer to the arena.
 * @return Pointer to the newly created child node.
 */
static Node* expand_node(Node *node, Arena *arena) {
    if (node->untried_moves.count == 0) return node; // Should not happen if checked before

    // Pop the last move from untried_moves (efficient)
    int idx = node->untried_moves.count - 1;
    Move move_to_try = node->untried_moves.moves[idx];
    node->untried_moves.count--;

    // Apply the move to get the new state
    GameState next_state = node->state;
    apply_move(&next_state, &move_to_try);

    // Create the child node
    Node *child = create_node(node, move_to_try, next_state, arena);

    // Resize children array using Arena (simple append logic)
    // Note: In a standard allocator, we would use realloc. 
    // Here we allocate a new array and copy. This is slightly inefficient but safe for Arena.
    // Optimization: Could allocate max_children upfront if MAX_MOVES is small.
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
static double calculate_ucb1(Node *child) {
    if (child->visits == 0) return 1e9; // Infinite value for unvisited nodes

    double win_rate = child->score / (double)child->visits;
    double exploration = UCB1_C * sqrt(log((double)child->parent->visits) / (double)child->visits);

    return win_rate + exploration;
}

/**
 * Selects the best child node to explore using UCB1.
 * Descends the tree until a node with untried moves or a terminal node is reached.
 */
static Node* select_promising_node(Node *root) {
    Node *current = root;
    
    // While fully expanded and not terminal
    while (!current->is_terminal && current->untried_moves.count == 0) {
        double best_ucb = -1.0;
        Node *best_child = NULL;

        for (int i = 0; i < current->num_children; i++) {
            double ucb = calculate_ucb1(current->children[i]);
            if (ucb > best_ucb) {
                best_ucb = ucb;
                best_child = current->children[i];
            }
        }
        
        if (best_child == NULL) break; // Should not happen
        current = best_child;
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
static Move pick_smart_move(const MoveList *list, const GameState *state, int use_lookahead) {
    if (list->count == 1) return list->moves[0];

    int best_score = -100000;
    int best_idx = 0;
    int random_tie_breaker = rand() % 10; 
    int us = state->current_player;

    for (int i = 0; i < list->count; i++) {
        Move m = list->moves[i];
        int score = random_tie_breaker;

        int from = m.path[0];
        int from_row = from / 8;

        int target_idx = (m.length == 0) ? 1 : m.length;
        int to = m.path[target_idx];
        int row = to / 8;
        int col = to % 8;

        // Promotion Bonus
        if (!m.is_lady_move) {
            if (row == 0 || row == 7) {
                score += WEIGHT_PROMOTION;
            }
            
            // Advancement Bonus
            // White moves UP (towards 7), Black moves DOWN (towards 0)
            int dist = (us == WHITE) ? (7 - row) : row;
            score += (7 - dist) * WEIGHT_ADVANCE;
        }

        // Safety Bonus (Edges)
        if (col == 0 || col == 7) {
            score += WEIGHT_SAFE_EDGE;
        }

        // Base Protection (Penalty)
        // Avoid moving from base rank (0 for White, 7 for Black) unless necessary
        if (!m.is_lady_move) {
             if ((us == WHITE && from_row == 0) || (us == BLACK && from_row == 7)) {
                 score -= WEIGHT_BASE_BREAK;
             }
        }

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
static int simulate_rollout(Node *node, MCTSConfig config) {
    GameState temp_state = node->state;
    
    // Check if the node itself is terminal
    if (node->is_terminal) {
        return (temp_state.current_player == WHITE) ? BLACK : WHITE;
    }

    int depth = 0;
    MoveList temp_moves;
    
    // Limit depth to avoid infinite loops
    while (depth < 200) { 
        generate_moves(&temp_state, &temp_moves);

        if (temp_moves.count == 0) {
            return (temp_state.current_player == WHITE) ? BLACK : WHITE;
        }
        
        if (temp_state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
            return -1; // Draw
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
            chosen_move = pick_smart_move(&temp_moves, &temp_state, config.use_lookahead);
        }

        apply_move(&temp_state, &chosen_move);
        depth++;
    }
    return -1; // Draw if depth limit reached
}

// ================================================================================================
//  BACKPROPAGATION
// ================================================================================================

/**
 * Propagates the simulation result back up the tree.
 * Updates visits and scores for all nodes in the path.
 */
static void backpropagate(Node *node, int winner) {
    while (node != NULL) {
        node->visits++;
        
        if (winner == -1) {
            node->score += DRAW_SCORE;
        } else if (winner == node->player_who_just_moved) {
            // If the winner is the player who just moved to create this node,
            // then this node represents a good state for that player.
            node->score += WIN_SCORE;
        } else {
            node->score += LOSS_SCORE;
        }
        node = node->parent;
    }
}

// ================================================================================================
//  PUBLIC API
// ================================================================================================

Node* mcts_create_root(GameState state, Arena *arena) {
    Move no_move = {0};
    return create_node(NULL, no_move, state, arena);
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
                 MCTSStats *stats) {
    clock_t start = clock();
    int iterations = 0;

    // Main MCTS Loop
    while ( (double)(clock() - start) / CLOCKS_PER_SEC < time_limit_seconds ) {
        // 1. Selection
        Node *leaf = select_promising_node(root);
        
        // 2. Expansion
        if (!leaf->is_terminal) {
            leaf = expand_node(leaf, arena); 
        }

        // 3. Simulation
        int winner = simulate_rollout(leaf, config);
        
        // 4. Backpropagation
        backpropagate(leaf, winner);
        
        iterations++;
    }

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

    return best_child->move_from_parent;
}
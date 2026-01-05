/**
 * mcts_utils.c - MCTS Utility Functions
 * 
 * Contains: mcts_create_root, tree traversal, policy extraction, diagnostics
 */

#include "dama/search/mcts.h"
#include "dama/search/mcts_tree.h"
#include "dama/neural/cnn.h"
#include "dama/common/params.h"
#include "dama/common/logging.h"
#include "dama/engine/movegen.h"
#include "dama/engine/game_view.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// =============================================================================
// ROOT NODE CREATION
// =============================================================================

Node* mcts_create_root(GameState state, Arena *arena, MCTSConfig config) {
    return mcts_create_root_with_history(state, arena, config, NULL);
}

Node* mcts_create_root_with_history(GameState state, Arena *arena, MCTSConfig config, Node *history_parent) {
    Move no_move = {0};
    return create_node(history_parent, no_move, state, arena, config);
}

// =============================================================================
// TREE TRAVERSAL
// =============================================================================

int get_tree_depth(const Node *node) {
    if (node->num_children == 0) return 0;
    
    int max_depth = 0;
    for (int i = 0; i < node->num_children; i++) {
        int child_depth = get_tree_depth(node->children[i]);
        if (child_depth > max_depth) max_depth = child_depth;
    }
    return max_depth + 1;
}

int get_tree_node_count(const Node *node) {
    if (!node) return 0;
    int count = 1; // This node
    for (int i = 0; i < node->num_children; i++) {
        count += get_tree_node_count(node->children[i]);
    }
    return count;
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

double mcts_get_avg_root_ucb(const Node *root, MCTSConfig config) {
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


// Only compile in debug builds to avoid memory allocation overhead in release
#ifndef NDEBUG
// Compare function for qsort (debug only)
static int compare_nodes_visits_desc(const void *a, const void *b) {
    Node *nodeA = *(Node **)a;
    Node *nodeB = *(Node **)b;
    return nodeB->visits - nodeA->visits;
}

void print_mcts_stats_sorted(const Node *root) {
    if (!root || root->num_children == 0) {
        printf("No children statistics available.\n");
        return;
    }

    Node **sorted_children = malloc(root->num_children * sizeof(Node*));
    if (!sorted_children) {
        log_warn("[mcts_utils] malloc failed in print_mcts_stats_sorted");
        return;
    }

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
#else
// Stub in release builds
void print_mcts_stats_sorted(const Node *root) {
    (void)root;
}
#endif

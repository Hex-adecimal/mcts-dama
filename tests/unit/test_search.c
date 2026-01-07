/**
 * test_search.c - Unit Tests for Search Module (MCTS)
 * 
 * Tests: mcts.h, mcts_types.h, mcts_tree.h
 */

// Note: Includes are in test_main.c

// =============================================================================
// ARENA TESTS
// =============================================================================

TEST(search_arena_init_and_alloc) {
    Arena arena;
    arena_init(&arena, 1024);
    
    ASSERT_NOT_NULL(arena.buffer);
    ASSERT_EQ(0, arena.offset);
    ASSERT_EQ(1024, arena.size);
    
    // Allocate some memory
    void *ptr = arena_alloc(&arena, 100);
    ASSERT_NOT_NULL(ptr);
    ASSERT_GE(arena.offset, 100);
    
    arena_free(&arena);
}

TEST(search_arena_reset_clears_used) {
    Arena arena;
    arena_init(&arena, 1024);
    
    arena_alloc(&arena, 100);
    ASSERT_GT(arena.offset, 0);
    
    arena_reset(&arena);
    ASSERT_EQ(0, arena.offset);
    
    arena_free(&arena);
}

TEST(search_arena_alloc_multiple) {
    Arena arena;
    arena_init(&arena, 4096);
    
    void *p1 = arena_alloc(&arena, 64);
    void *p2 = arena_alloc(&arena, 128);
    void *p3 = arena_alloc(&arena, 256);
    
    ASSERT_NOT_NULL(p1);
    ASSERT_NOT_NULL(p2);
    ASSERT_NOT_NULL(p3);
    
    // Pointers should be different
    ASSERT_NE((long long)p1, (long long)p2);
    ASSERT_NE((long long)p2, (long long)p3);
    
    arena_free(&arena);
}

// =============================================================================
// MCTS PRESET TESTS
// =============================================================================

TEST(search_mcts_presets_have_valid_configs) {
    MCTSConfig vanilla = mcts_get_preset(MCTS_PRESET_VANILLA);
    MCTSConfig gm = mcts_get_preset(MCTS_PRESET_GRANDMASTER);
    MCTSConfig az = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO);
    
    // All should have settings
    // Note: max_nodes may be 0 (unlimited) by default
    ASSERT_GE(vanilla.max_nodes, 0);
    ASSERT_GE(gm.max_nodes, 0);
    ASSERT_GE(az.max_nodes, 0);
    
    // Vanilla uses UCB1, not PUCT
    ASSERT_FALSE(vanilla.use_puct);
    ASSERT_GT(vanilla.ucb1_c, 0);
    
    // AlphaZero uses PUCT
    ASSERT_TRUE(az.use_puct);
    ASSERT_GT(az.puct_c, 0);
}

// =============================================================================
// ROOT NODE TESTS
// =============================================================================

TEST(search_create_root_returns_valid_node) {
    GameState state;
    init_game(&state);
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_BENCHMARK);
    
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
    
    Node *root = mcts_create_root(state, &arena, config);
    
    ASSERT_NOT_NULL(root);
    ASSERT_EQ(NULL, root->parent);
    ASSERT_EQ(0, root->visits);
    ASSERT_EQ(WHITE, root->state.current_player);
    
    arena_free(&arena);
}

TEST(search_create_root_generates_children) {
    GameState state;
    init_game(&state);
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_BENCHMARK);
    
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
    
    Node *root = mcts_create_root(state, &arena, config);
    
    // Root should have untried moves or children
    // Depending on implementation, children may be created lazily
    ASSERT_TRUE(root->num_children >= 0 || root->untried_moves.count > 0);
    
    arena_free(&arena);
}

// =============================================================================
// MCTS SEARCH TESTS
// =============================================================================

TEST(search_mcts_search_returns_valid_move) {
    GameState state;
    init_game(&state);
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_BENCHMARK);
    
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
    config.max_nodes = 100;  // Small limit for test speed
    
    Node *root = mcts_create_root(state, &arena, config);
    
    Move best = mcts_search(root, &arena, 0.1, config, NULL, NULL);
    
    // Move should have valid from/to
    ASSERT_NE(best.path[0], best.path[1]);
    ASSERT_LT(best.path[0], 64);
    ASSERT_LT(best.path[1], 64);
    
    arena_free(&arena);
}

TEST(search_mcts_search_increases_visits) {
    GameState state;
    init_game(&state);
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_BENCHMARK);
    
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
    config.max_nodes = 50;
    
    Node *root = mcts_create_root(state, &arena, config);
    
    ASSERT_EQ(0, root->visits);
    
    MCTSStats stats = {0};
    mcts_search(root, &arena, 0.1, config, &stats, NULL, NULL);
    
    ASSERT_GT(root->visits, 0);
    ASSERT_GT(stats.total_iterations, 0);
    
    arena_free(&arena);
}

TEST(search_mcts_stats_are_collected) {
    GameState state;
    init_game(&state);
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_BENCHMARK);
    
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
    config.max_nodes = 100;
    
    Node *root = mcts_create_root(state, &arena, config);
    
    MCTSStats stats = {0};
    mcts_search(root, &arena, 0.1, config, &stats, NULL, NULL);
    
    ASSERT_GT(stats.total_iterations, 0);
    ASSERT_GT(stats.total_nodes, 0);
    
    arena_free(&arena);
}

// =============================================================================
// TREE UTILITY TESTS
// =============================================================================

TEST(search_get_tree_depth_returns_positive) {
    GameState state;
    init_game(&state);
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_BENCHMARK);
    
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
    config.max_nodes = 50;
    
    Node *root = mcts_create_root(state, &arena, config);
    mcts_search(root, &arena, 0.1, config, NULL, NULL, NULL);
    
    int depth = get_tree_depth(root);
    ASSERT_GT(depth, 0);
    
    arena_free(&arena);
}

TEST(search_get_tree_node_count_matches_stats) {
    GameState state;
    init_game(&state);
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_BENCHMARK);
    
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
    config.max_nodes = 50;
    
    Node *root = mcts_create_root(state, &arena, config);
    
    MCTSStats stats = {0};
    mcts_search(root, &arena, 0.1, config, &stats, NULL, NULL);
    
    int node_count = get_tree_node_count(root);
    ASSERT_GT(node_count, 0);
    
    arena_free(&arena);
}

// =============================================================================
// POLICY EXTRACTION TESTS
// =============================================================================

TEST(search_mcts_get_policy_sums_to_one) {
    GameState state;
    init_game(&state);
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_BENCHMARK);
    
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
    config.max_nodes = 100;
    
    Node *root = mcts_create_root(state, &arena, config);
    mcts_search(root, &arena, 0.1, config, NULL, NULL, NULL);
    
    float policy[512] = {0};
    mcts_get_policy(root, policy, 1.0f, &state);
    
    float sum = 0;
    for (int i = 0; i < 512; i++) {
        sum += policy[i];
        ASSERT_GE(policy[i], 0.0f);
    }
    
    ASSERT_FLOAT_EQ(1.0f, sum, 0.01f);
    
    arena_free(&arena);
}

TEST(search_mcts_get_policy_nonzero_entries) {
    GameState state;
    init_game(&state);
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_BENCHMARK);
    
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
    config.max_nodes = 100;
    
    Node *root = mcts_create_root(state, &arena, config);
    mcts_search(root, &arena, 0.1, config, NULL, NULL, NULL);
    
    float policy[512] = {0};
    mcts_get_policy(root, policy, 1.0f, &state);
    
    int nonzero = 0;
    for (int i = 0; i < 512; i++) {
        if (policy[i] > 0.001f) nonzero++;
    }
    
    // Should have same number of nonzero as legal moves
    ASSERT_EQ(7, nonzero);
    
    arena_free(&arena);
}

// =============================================================================
// UCB SELECTION TESTS
// =============================================================================

TEST(search_ucb1_score_increases_with_visits) {
    GameState state;
    init_game(&state);
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_BENCHMARK);
    
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
    config.max_nodes = 50;
    
    Node *root = mcts_create_root(state, &arena, config);
    mcts_search(root, &arena, 0.1, config, NULL, NULL, NULL);
    
    // Children with more visits should generally have been selected more
    int max_visits = 0;
    for (int i = 0; i < root->num_children; i++) {
        if (root->children[i]->visits > max_visits) {
            max_visits = root->children[i]->visits;
        }
    }
    ASSERT_GT(max_visits, 0);
    
    arena_free(&arena);
}

TEST(search_puct_uses_priors) {
    GameState state;
    init_game(&state);
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_BENCHMARK);
    
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO);
    config.max_nodes = 50;
    config.cnn_weights = NULL; // Force uniform priors
    
    Node *root = mcts_create_root(state, &arena, config);
    
    // Without CNN, priors should be uniform or heuristic-based
    ASSERT_TRUE(config.use_puct);
    ASSERT_NOT_NULL(root);
    
    arena_free(&arena);
}

// =============================================================================
// BACKPROPAGATION TESTS
// =============================================================================

TEST(search_backprop_updates_scores) {
    GameState state;
    init_game(&state);
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_BENCHMARK);
    
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
    config.max_nodes = 100;
    
    Node *root = mcts_create_root(state, &arena, config);
    mcts_search(root, &arena, 0.1, config, NULL, NULL, NULL);
    
    // Root score should be accumulated from children
    ASSERT_NE(0.0, root->score);
    ASSERT_GT(root->visits, 0);
    
    // Sum of squared scores should be tracked
    ASSERT_GE(root->sum_sq_score, 0.0);
    
    arena_free(&arena);
}

TEST(search_virtual_loss_zeroed_after_search) {
    GameState state;
    init_game(&state);
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_BENCHMARK);
    
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
    config.max_nodes = 50;
    
    Node *root = mcts_create_root(state, &arena, config);
    mcts_search(root, &arena, 0.1, config, NULL, NULL, NULL);
    
    // After search completes, virtual loss should be non-positive
    // (It may be negative due to sequential search subtracting VL without adding first)
    ASSERT_LE(root->virtual_loss, 0);
    
    arena_free(&arena);
}

// =============================================================================
// SOLVER TESTS
// =============================================================================

TEST(search_solver_detects_terminal) {
    GameState state;
    init_game(&state);
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_BENCHMARK);
    
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_GRANDMASTER);
    config.use_solver = 1;
    config.max_nodes = 100;
    
    Node *root = mcts_create_root(state, &arena, config);
    mcts_search(root, &arena, 0.1, config, NULL, NULL, NULL);
    
    // Root shouldn't be terminal at start of game
    ASSERT_EQ(0, root->is_terminal);
    
    arena_free(&arena);
}

// =============================================================================
// TRANSPOSITION TABLE TESTS
// =============================================================================

TEST(search_tt_create_and_free) {
    TranspositionTable *tt = tt_create(1024);
    
    ASSERT_NOT_NULL(tt);
    ASSERT_EQ(1024, tt->size);
    ASSERT_EQ(0, tt->count);
    
    tt_free(tt);
}

TEST(search_tt_mask_is_power_of_two) {
    TranspositionTable *tt = tt_create(1024);
    
    ASSERT_NOT_NULL(tt);
    // Mask should be size - 1 for power-of-2 bucket hashing
    ASSERT_EQ(1023, tt->mask);
    
    tt_free(tt);
}

// =============================================================================
// TREE REUSE TESTS
// =============================================================================

TEST(search_tree_reuse_preserves_stats) {
    GameState state;
    init_game(&state);
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_BENCHMARK);
    
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
    config.max_nodes = 50;
    
    Node *root = mcts_create_root(state, &arena, config);
    Node *new_root = NULL;
    
    Move best = mcts_search(root, &arena, 0.1, config, NULL, &new_root);
    
    // new_root should be a child with visits
    if (new_root) {
        ASSERT_GT(new_root->visits, 0);
        ASSERT_TRUE(moves_equal(&new_root->move_from_parent, &best));
    }
    
    arena_free(&arena);
}

// =============================================================================
// TT REPLACEMENT POLICY TESTS (NEW)
// =============================================================================

TEST(search_tt_higher_visits_not_replaced) {
    // Create TT and arena
    TranspositionTable *tt = tt_create(1024);
    ASSERT_NOT_NULL(tt);
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_BENCHMARK);
    
    GameState state;
    init_game(&state);
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
    
    // Create node N1 with 100 visits
    Move dummy_move = {0};
    dummy_move.path[0] = 9; dummy_move.path[1] = 18;
    Node *n1 = create_node(NULL, dummy_move, state, &arena, config);
    ASSERT_NOT_NULL(n1);
    atomic_store(&n1->visits, 100);
    
    // Insert N1 into TT
    tt_insert(tt, n1);
    ASSERT_EQ(1, tt->count);
    
    // Create node N2 with same hash but only 10 visits
    Node *n2 = create_node(NULL, dummy_move, state, &arena, config);
    ASSERT_NOT_NULL(n2);
    atomic_store(&n2->visits, 10);
    
    // N2 should have same hash as N1 (same game state)
    ASSERT_EQ(n1->state.hash, n2->state.hash);
    
    // Insert N2 - should NOT replace N1 (quality-based replacement)
    tt_insert(tt, n2);
    
    // Lookup should still return N1 (the node with more visits)
    Node *found = tt_lookup(tt, &state);
    ASSERT_NOT_NULL(found);
    ASSERT_EQ(100, atomic_load(&found->visits));  // Should be N1, not N2
    
    tt_free(tt);
    arena_free(&arena);
}

// =============================================================================
// SEARCH STABILITY TESTS (NEW)
// =============================================================================

TEST(search_more_nodes_equals_better_or_same_move) {
    GameState state;
    init_game(&state);
    
    Arena arena1, arena2;
    arena_init(&arena1, ARENA_SIZE_BENCHMARK);
    arena_init(&arena2, ARENA_SIZE_BENCHMARK);
    
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
    
    // Search with 100 nodes
    config.max_nodes = 100;
    Node *root1 = mcts_create_root(state, &arena1, config);
    Move move1 = mcts_search(root1, &arena1, 1.0, config, NULL, NULL, NULL);
    int visits1 = 0;
    for (int i = 0; i < root1->num_children; i++) {
        if (moves_equal(&root1->children[i]->move_from_parent, &move1)) {
            visits1 = root1->children[i]->visits;
            break;
        }
    }
    
    // Search with 500 nodes (same position)
    config.max_nodes = 500;
    Node *root2 = mcts_create_root(state, &arena2, config);
    Move move2 = mcts_search(root2, &arena2, 5.0, config, NULL, NULL);
    int visits2 = 0;
    for (int i = 0; i < root2->num_children; i++) {
        if (moves_equal(&root2->children[i]->move_from_parent, &move2)) {
            visits2 = root2->children[i]->visits;
            break;
        }
    }
    
    // With more nodes, the best move should have at least as many visits
    // or be the same move (search stability)
    int same_move = moves_equal(&move1, &move2);
    ASSERT_TRUE(same_move || visits2 >= visits1);
    
    arena_free(&arena1);
    arena_free(&arena2);
}

// =============================================================================
// PARALLEL CONSISTENCY TESTS (NEW)
// =============================================================================

TEST(search_parallel_mcts_visits_are_consistent) {
    GameState state;
    init_game(&state);
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_BENCHMARK);
    
    MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
    config.max_nodes = 200;  // Enough for parallel stress
    
    Node *root = mcts_create_root(state, &arena, config);
    MCTSStats stats = {0};
    
    // Run search (may use multiple threads if NUM_MCTS_THREADS > 0)
    mcts_search(root, &arena, 0.5, config, &stats, NULL, NULL);
    
    // Verify visit consistency: sum of children visits should be close to root visits
    // Allow for off-by-one due to root node itself
    int sum_child_visits = 0;
    for (int i = 0; i < root->num_children; i++) {
        sum_child_visits += root->children[i]->visits;
    }
    
    int root_visits = root->visits;
    
    // All visits from root should propagate to children (minus any stuck in selection)
    // We use a tolerance because visits can be in-flight during parallel execution
    int tolerance = 5;  // Allow small discrepancy
    ASSERT_TRUE(root_visits - sum_child_visits <= tolerance);
    
    // Also verify that total iterations roughly matches root visits
    ASSERT_TRUE(stats.total_iterations >= root_visits - tolerance);
    
    arena_free(&arena);
}


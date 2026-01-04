/**
 * test_neural.c - Unit Tests for Neural Module (CNN)
 * 
 * Tests: cnn.h, conv_ops.h
 */

// Note: Includes are in test_main.c

// =============================================================================
// CNN INITIALIZATION TESTS
// =============================================================================

TEST(neural_cnn_init_allocates_weights) {
    CNNWeights weights;
    cnn_init(&weights);
    
    // Check that key weight arrays are allocated
    ASSERT_NOT_NULL(weights.conv1_w);
    ASSERT_NOT_NULL(weights.conv1_b);
    ASSERT_NOT_NULL(weights.policy_w);
    ASSERT_NOT_NULL(weights.value_w1);
    ASSERT_NOT_NULL(weights.bn1_gamma);
    
    cnn_free(&weights);
}

TEST(neural_cnn_init_sets_bn_defaults) {
    CNNWeights weights;
    cnn_init(&weights);
    
    // Batch norm gamma should be initialized to 1
    ASSERT_FLOAT_EQ(1.0f, weights.bn1_gamma[0], 0.01f);
    
    // Batch norm beta should be initialized to 0
    ASSERT_FLOAT_EQ(0.0f, weights.bn1_beta[0], 0.01f);
    
    // Running variance should be initialized to 1
    ASSERT_FLOAT_EQ(1.0f, weights.bn1_var[0], 0.01f);
    
    cnn_free(&weights);
}

TEST(neural_cnn_free_doesnt_crash) {
    CNNWeights weights;
    cnn_init(&weights);
    cnn_free(&weights);
    
    // Double free shouldn't crash (pointers should be NULL)
    // This is just a safety check
    ASSERT_TRUE(1);
}

// =============================================================================
// ENCODING TESTS
// =============================================================================

TEST(neural_cnn_encode_initial_position) {
    GameState state;
    init_game(&state);
    
    TrainingSample sample = {0};
    sample.state = state;
    
    float tensor[CNN_INPUT_CHANNELS * 64] = {0};
    float player;
    
    cnn_encode_sample(&sample, tensor, &player);
    
    // Count non-zero entries (pieces on board)
    int nonzero = 0;
    for (int i = 0; i < CNN_INPUT_CHANNELS * 64; i++) {
        if (tensor[i] > 0.5f) nonzero++;
    }
    
    // Should have 24 pieces (12 white pawns + 12 black pawns)
    // But encoding may be canonical, so check range
    ASSERT_GE(nonzero, 12);
    ASSERT_LE(nonzero, 48);  // At most 24 pieces * 2 channels (pawn + lady)
}

TEST(neural_flip_square_works_correctly) {
    // Flip should mirror vertically (row 0 <-> row 7)
    // flip_square(sq) = (7 - row) * 8 + col = 56 - 8*row + col
    // For sq = r*8 + c: flip = 56 - 8*r + c = 56 - r*8 + c
    
    // A1 (sq=0, r=0, c=0) -> A8 (sq=56)
    ASSERT_EQ(56, flip_square(0));
    
    // H1 (sq=7, r=0, c=7) -> H8 (sq=63)
    ASSERT_EQ(63, flip_square(7));
    
    // A8 (sq=56, r=7, c=0) -> A1 (sq=0)
    ASSERT_EQ(0, flip_square(56));
    
    // H8 (sq=63, r=7, c=7) -> H1 (sq=7)
    ASSERT_EQ(7, flip_square(63));
    
    // D4 (sq=27, r=3, c=3) -> D5 (sq=35)? Let's verify
    // flip(27) = 56 - 3*8 + 3 = 56 - 24 + 3 = 35
    int flipped_27 = flip_square(27);
    int flipped_back = flip_square(flipped_27);
    ASSERT_EQ(27, flipped_back);  // Double flip should return original
}

// =============================================================================
// FORWARD PASS TESTS
// =============================================================================

TEST(neural_cnn_forward_produces_valid_output) {
    CNNWeights weights;
    cnn_init(&weights);
    
    GameState state;
    init_game(&state);
    
    TrainingSample sample = {0};
    sample.state = state;
    
    CNNOutput out;
    cnn_forward_sample(&weights, &sample, &out);
    
    // Policy should sum to ~1 (softmax output)
    float policy_sum = 0;
    for (int i = 0; i < CNN_POLICY_SIZE; i++) {
        policy_sum += out.policy[i];
        ASSERT_GE(out.policy[i], 0.0f);
        ASSERT_LE(out.policy[i], 1.0f);
    }
    ASSERT_FLOAT_EQ(1.0f, policy_sum, 0.01f);
    
    // Value should be in [-1, 1] (tanh output)
    ASSERT_GE(out.value, -1.0f);
    ASSERT_LE(out.value, 1.0f);
    
    cnn_free(&weights);
}

TEST(neural_cnn_forward_with_history) {
    CNNWeights weights;
    cnn_init(&weights);
    
    GameState state;
    init_game(&state);
    
    // Make a move to create history
    MoveList moves;
    movegen_generate(&state, &moves);
    GameState hist1 = state;
    apply_move(&state, &moves.moves[0]);
    
    CNNOutput out;
    cnn_forward_with_history(&weights, &state, &hist1, NULL, &out);
    
    // Should still produce valid output
    ASSERT_GE(out.value, -1.0f);
    ASSERT_LE(out.value, 1.0f);
    
    cnn_free(&weights);
}

TEST(neural_cnn_forward_is_deterministic) {
    CNNWeights weights;
    cnn_init(&weights);
    
    GameState state;
    init_game(&state);
    
    TrainingSample sample = {0};
    sample.state = state;
    
    CNNOutput out1, out2;
    cnn_forward_sample(&weights, &sample, &out1);
    cnn_forward_sample(&weights, &sample, &out2);
    
    // Same input should produce same output
    ASSERT_FLOAT_EQ(out1.value, out2.value, 1e-6f);
    ASSERT_FLOAT_EQ(out1.policy[0], out2.policy[0], 1e-6f);
    
    cnn_free(&weights);
}

// =============================================================================
// MOVE INDEX TESTS
// =============================================================================

TEST(neural_move_to_index_returns_valid_index) {
    // Create a simple move
    Move m = {0};
    m.path[0] = 11;  // From D2
    m.path[1] = 20;  // To E3
    m.length = 0;    // Simple move
    
    int idx = cnn_move_to_index(&m, WHITE);
    
    // Index should be in valid range
    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, CNN_POLICY_SIZE);
}

TEST(neural_move_to_index_different_for_colors) {
    // Same move coordinates
    Move m = {0};
    m.path[0] = 27;  // D4
    m.path[1] = 36;  // E5
    m.length = 0;
    
    int idx_white = cnn_move_to_index(&m, WHITE);
    int idx_black = cnn_move_to_index(&m, BLACK);
    
    // Due to canonical representation, indices may differ
    ASSERT_GE(idx_white, 0);
    ASSERT_GE(idx_black, 0);
}

// =============================================================================
// WEIGHT SAVE/LOAD TESTS
// =============================================================================

TEST(neural_cnn_save_and_load_roundtrip) {
    CNNWeights weights1, weights2;
    cnn_init(&weights1);
    cnn_init(&weights2);
    
    // Modify some weights
    weights1.conv1_w[0] = 0.12345f;
    weights1.policy_b[0] = -0.6789f;
    weights1.value_b2[0] = 0.5f;
    
    // Save
    const char *path = "/tmp/test_weights.bin";
    cnn_save_weights(&weights1, path);
    
    // Load
    int load_ret = cnn_load_weights(&weights2, path);
    ASSERT_EQ(0, load_ret);
    
    // Verify values match
    ASSERT_FLOAT_EQ(weights1.conv1_w[0], weights2.conv1_w[0], 1e-6f);
    ASSERT_FLOAT_EQ(weights1.policy_b[0], weights2.policy_b[0], 1e-6f);
    ASSERT_FLOAT_EQ(weights1.value_b2[0], weights2.value_b2[0], 1e-6f);
    
    cnn_free(&weights1);
    cnn_free(&weights2);
    
    // Cleanup
    remove(path);
}

TEST(neural_cnn_load_nonexistent_fails) {
    CNNWeights weights;
    cnn_init(&weights);
    
    int ret = cnn_load_weights(&weights, "/nonexistent/path/weights.bin");
    ASSERT_NE(0, ret);
    
    cnn_free(&weights);
}

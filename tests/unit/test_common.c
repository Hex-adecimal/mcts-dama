/**
 * test_common.c - Unit Tests for Common Module
 * 
 * Tests: rng.h, params.h, logging.h
 */

// Note: Includes are in test_main.c

// =============================================================================
// RNG TESTS
// =============================================================================

TEST(common_rng_seed_sets_state) {
    RNG rng;
    rng_seed(&rng, 12345);
    
    ASSERT_EQ(12345, rng.state);
}

TEST(common_rng_seed_zero_becomes_one) {
    RNG rng;
    rng_seed(&rng, 0);
    
    // Zero seed should become 1 to avoid degenerate sequence
    ASSERT_EQ(1, rng.state);
}

TEST(common_rng_u32_changes_state) {
    RNG rng;
    rng_seed(&rng, 42);
    
    uint32_t initial = rng.state;
    uint32_t val = rng_u32(&rng);
    
    ASSERT_NE(initial, rng.state);
    ASSERT_NE(0, val);  // Extremely unlikely to be 0
}

TEST(common_rng_u32_is_deterministic) {
    RNG rng1, rng2;
    rng_seed(&rng1, 12345);
    rng_seed(&rng2, 12345);
    
    for (int i = 0; i < 100; i++) {
        ASSERT_EQ(rng_u32(&rng1), rng_u32(&rng2));
    }
}

TEST(common_rng_u32_different_seeds_differ) {
    RNG rng1, rng2;
    rng_seed(&rng1, 1);
    rng_seed(&rng2, 2);
    
    // First values should differ
    ASSERT_NE(rng_u32(&rng1), rng_u32(&rng2));
}

TEST(common_rng_f32_in_range) {
    RNG rng;
    rng_seed(&rng, 99999);
    
    for (int i = 0; i < 1000; i++) {
        float val = rng_f32(&rng);
        ASSERT_GE(val, 0.0f);
        ASSERT_LE(val, 1.0f);
    }
}

TEST(common_rng_f32_distribution) {
    RNG rng;
    rng_seed(&rng, 54321);
    
    int below_half = 0;
    int above_half = 0;
    
    for (int i = 0; i < 10000; i++) {
        float val = rng_f32(&rng);
        if (val < 0.5f) below_half++;
        else above_half++;
    }
    
    // Should be roughly 50/50 (allow 10% tolerance)
    float ratio = (float)below_half / (float)above_half;
    ASSERT_GT(ratio, 0.8f);
    ASSERT_LT(ratio, 1.2f);
}

TEST(common_rng_gamma_positive) {
    RNG rng;
    rng_seed(&rng, 77777);
    
    for (int i = 0; i < 100; i++) {
        float val = rng_gamma(&rng, 1.0f);
        ASSERT_GT(val, 0.0f);
    }
}

TEST(common_rng_gamma_alpha_less_than_one) {
    RNG rng;
    rng_seed(&rng, 88888);
    
    // Alpha < 1 uses the recursive method
    for (int i = 0; i < 100; i++) {
        float val = rng_gamma(&rng, 0.5f);
        ASSERT_GT(val, 0.0f);
    }
}

TEST(common_rng_gamma_mean_approximately_alpha) {
    RNG rng;
    rng_seed(&rng, 11111);
    
    float alpha = 2.0f;
    float sum = 0;
    int n = 10000;
    
    for (int i = 0; i < n; i++) {
        sum += rng_gamma(&rng, alpha);
    }
    
    float mean = sum / n;
    
    // Gamma(alpha, 1) has mean = alpha
    // Allow 10% tolerance
    ASSERT_GT(mean, alpha * 0.9f);
    ASSERT_LT(mean, alpha * 1.1f);
}

// =============================================================================
// PARAMS TESTS
// =============================================================================

TEST(common_params_time_limits_positive) {
    ASSERT_GT(TIME_LOW, 0);
    ASSERT_GT(TIME_HIGH, 0);
    ASSERT_GT(TIME_HIGH, TIME_LOW);
}

TEST(common_params_arena_size_reasonable) {
    // Should be big enough for MCTS trees
    ASSERT_GT(ARENA_SIZE, 1000000);  // At least 1MB
    ASSERT_GT(ARENA_SIZE_TOURNAMENT, 100000);  // At least 100KB
}

TEST(common_params_cnn_defaults_valid) {
    ASSERT_GT(CNN_DEFAULT_EPOCHS, 0);
    ASSERT_GT(CNN_DEFAULT_BATCH_SIZE, 0);
    ASSERT_GT(CNN_POLICY_LR, 0);
    ASSERT_GT(CNN_VALUE_LR, 0);
    ASSERT_GE(CNN_DEFAULT_MOMENTUM, 0);
    ASSERT_LT(CNN_DEFAULT_MOMENTUM, 1);
}

TEST(common_params_mcts_weights_defined) {
    // Check that MCTS evaluation weights are defined
    ASSERT_TRUE(W_CAPTURE != 0 || W_PROMOTION != 0 || W_CENTER != 0);
}

// =============================================================================
// UTILITY TESTS
// =============================================================================

TEST(common_bit_operations_correct) {
    // Test some common bit patterns
    uint64_t bb = 0;
    
    // Set bit 0
    bb |= (1ULL << 0);
    ASSERT_EQ(1ULL, bb);
    
    // Set bit 63
    bb |= (1ULL << 63);
    ASSERT_EQ(0x8000000000000001ULL, bb);
    
    // Clear bit 0
    bb &= ~(1ULL << 0);
    ASSERT_EQ(0x8000000000000000ULL, bb);
}

TEST(common_popcount_works) {
    ASSERT_EQ(0, __builtin_popcountll(0ULL));
    ASSERT_EQ(1, __builtin_popcountll(1ULL));
    ASSERT_EQ(1, __builtin_popcountll(0x8000000000000000ULL));
    ASSERT_EQ(64, __builtin_popcountll(0xFFFFFFFFFFFFFFFFULL));
    ASSERT_EQ(8, __builtin_popcountll(0xFF));
}

// =============================================================================
// ERROR CODES TESTS
// =============================================================================

TEST(common_error_codes_defined) {
    // Verify error codes are properly defined
    ASSERT_EQ(0, ERR_OK);
    ASSERT_EQ(-1, ERR_NULL_PTR);
    ASSERT_EQ(-2, ERR_MEMORY);
    ASSERT_EQ(-3, ERR_FILE_OPEN);
    ASSERT_EQ(-4, ERR_FILE_FORMAT);
    ASSERT_EQ(-5, ERR_VERSION);
    ASSERT_EQ(-6, ERR_INVALID_ARG);
    ASSERT_EQ(-7, ERR_NOT_IMPL);
}

// =============================================================================
// DEBUG ASSERT TESTS
// =============================================================================

TEST(common_debug_dbg_not_null_passes) {
    // This should not log anything since pointer is valid
    int value = 42;
    int *ptr = &value;
    DBG_NOT_NULL(ptr);
    ASSERT_EQ(42, *ptr);
}

TEST(common_debug_dbg_valid_sq_passes) {
    // Valid squares 0-63 should pass
    DBG_VALID_SQ(0);
    DBG_VALID_SQ(31);
    DBG_VALID_SQ(63);
}

TEST(common_debug_dbg_valid_color_passes) {
    // Valid colors 0 (WHITE) and 1 (BLACK) should pass
    DBG_VALID_COLOR(0);
    DBG_VALID_COLOR(1);
}


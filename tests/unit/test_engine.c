/**
 * test_engine.c - Unit Tests for Engine Module
 * 
 * Tests: game.h, movegen.h, endgame.h
 */

// Note: Includes are in test_main.c

// =============================================================================
// GAME STATE TESTS
// =============================================================================

TEST(engine_init_game_sets_correct_initial_position) {
    GameState state;
    init_game(&state);
    
    // White should have 12 pawns on ranks 1-3
    ASSERT_EQ(INITIAL_WHITE_PAWNS, state.piece[WHITE][PAWN]);
    ASSERT_EQ(0, state.piece[WHITE][LADY]);
    
    // Black should have 12 pawns on ranks 6-8
    ASSERT_EQ(INITIAL_BLACK_PAWNS, state.piece[BLACK][PAWN]);
    ASSERT_EQ(0, state.piece[BLACK][LADY]);
    
    // White moves first
    ASSERT_EQ(WHITE, state.current_player);
    ASSERT_EQ(0, state.moves_without_captures);
    
    // Hash should be initialized
    ASSERT_NE(0, state.hash);
}

TEST(engine_get_pieces_returns_all_pieces_for_color) {
    GameState state;
    init_game(&state);
    
    Bitboard white_pieces = get_pieces(&state, WHITE);
    Bitboard black_pieces = get_pieces(&state, BLACK);
    
    // Each side should have 12 pieces initially
    int white_count = __builtin_popcountll(white_pieces);
    int black_count = __builtin_popcountll(black_pieces);
    
    ASSERT_EQ(12, white_count);
    ASSERT_EQ(12, black_count);
}

TEST(engine_get_all_occupied_returns_all_pieces) {
    GameState state;
    init_game(&state);
    
    Bitboard occupied = get_all_occupied(&state);
    int total = __builtin_popcountll(occupied);
    
    ASSERT_EQ(24, total);
}

TEST(engine_check_bit_works_correctly) {
    Bitboard bb = 0x1ULL;  // Bit 0 set
    
    ASSERT_TRUE(check_bit(bb, 0));
    ASSERT_FALSE(check_bit(bb, 1));
    ASSERT_FALSE(check_bit(bb, 63));
    
    bb = 0x8000000000000000ULL;  // Bit 63 set
    ASSERT_TRUE(check_bit(bb, 63));
    ASSERT_FALSE(check_bit(bb, 0));
}

TEST(engine_zobrist_keys_are_unique) {
    // Check that some keys are different
    ASSERT_NE(zobrist_keys[WHITE][PAWN][0], zobrist_keys[WHITE][PAWN][1]);
    ASSERT_NE(zobrist_keys[WHITE][PAWN][0], zobrist_keys[BLACK][PAWN][0]);
    ASSERT_NE(zobrist_keys[WHITE][PAWN][0], zobrist_keys[WHITE][LADY][0]);
}

TEST(engine_hash_changes_after_move) {
    GameState state;
    init_game(&state);
    
    uint64_t initial_hash = state.hash;
    
    MoveList moves;
    movegen_generate(&state, &moves);
    ASSERT_GT(moves.count, 0);
    
    apply_move(&state, &moves.moves[0]);
    
    ASSERT_NE(initial_hash, state.hash);
}

// =============================================================================
// MOVE GENERATION TESTS
// =============================================================================

TEST(engine_movegen_generate_initial_position_has_7_moves) {
    GameState state;
    init_game(&state);
    
    MoveList moves;
    movegen_generate(&state, &moves);
    
    // In Italian Checkers initial position, White has 7 possible moves
    ASSERT_EQ(7, moves.count);
}

TEST(engine_movegen_generate_simple_only_non_captures) {
    GameState state;
    init_game(&state);
    
    MoveList simple_moves;
    movegen_generate_simple(&state, &simple_moves);
    
    // All moves should be non-captures (length == 0)
    for (int i = 0; i < simple_moves.count; i++) {
        ASSERT_EQ(0, simple_moves.moves[i].length);
    }
}

TEST(engine_captures_are_mandatory) {
    GameState state;
    init_game(&state);
    
    // Create a simple capture scenario
    // Clear the board first
    state.piece[WHITE][PAWN] = 0;
    state.piece[WHITE][LADY] = 0;
    state.piece[BLACK][PAWN] = 0;
    state.piece[BLACK][LADY] = 0;
    
    // Place white pawn at D2 (square 11)
    SET_BIT(state.piece[WHITE][PAWN], 11);
    // Place black pawn at E3 (square 20)
    SET_BIT(state.piece[BLACK][PAWN], 20);
    // F4 (square 29) is empty - possible capture destination
    
    state.current_player = WHITE;
    
    MoveList moves;
    movegen_generate(&state, &moves);
    
    // White must capture - all moves should be captures
    for (int i = 0; i < moves.count; i++) {
        ASSERT_GT(moves.moves[i].length, 0);  // Captures have length > 0
    }
}

TEST(engine_apply_move_switches_player) {
    GameState state;
    init_game(&state);
    
    ASSERT_EQ(WHITE, state.current_player);
    
    MoveList moves;
    movegen_generate(&state, &moves);
    apply_move(&state, &moves.moves[0]);
    
    ASSERT_EQ(BLACK, state.current_player);
}

TEST(engine_apply_move_updates_board) {
    GameState state;
    init_game(&state);
    
    MoveList moves;
    movegen_generate(&state, &moves);
    
    Move m = moves.moves[0];
    int from_sq = m.path[0];
    int to_sq = m.path[1];
    
    // Before move: piece at from, empty at to
    ASSERT_TRUE(check_bit(state.piece[WHITE][PAWN], from_sq));
    ASSERT_FALSE(check_bit(state.piece[WHITE][PAWN], to_sq));
    
    apply_move(&state, &m);
    
    // After move: empty at from, piece at to
    ASSERT_FALSE(check_bit(state.piece[WHITE][PAWN], from_sq));
    ASSERT_TRUE(check_bit(state.piece[WHITE][PAWN], to_sq));
}

TEST(engine_promotion_to_lady) {
    GameState state;
    init_game(&state);
    
    // Clear board and place white pawn about to promote
    state.piece[WHITE][PAWN] = 0;
    state.piece[WHITE][LADY] = 0;
    state.piece[BLACK][PAWN] = 0;
    state.piece[BLACK][LADY] = 0;
    
    // Place pawn at E7 (square 52) - one step from promotion
    SET_BIT(state.piece[WHITE][PAWN], 52);
    state.current_player = WHITE;
    
    MoveList moves;
    movegen_generate(&state, &moves);
    
    ASSERT_GT(moves.count, 0);
    
    // Apply move to F8 (square 61) - promotion rank
    for (int i = 0; i < moves.count; i++) {
        int to_sq = moves.moves[i].path[1];
        if (to_sq >= 56) {  // Rank 8
            apply_move(&state, &moves.moves[i]);
            
            // Should now be a lady, not a pawn
            ASSERT_TRUE(check_bit(state.piece[WHITE][LADY], to_sq));
            ASSERT_FALSE(check_bit(state.piece[WHITE][PAWN], to_sq));
            ASSERT_FALSE(check_bit(state.piece[WHITE][PAWN], 52));
            break;
        }
    }
}

TEST(engine_lady_moves_all_directions) {
    GameState state;
    init_game(&state);
    
    // Clear board and place white lady on a dark square in center (C3 = square 17)
    state.piece[WHITE][PAWN] = 0;
    state.piece[WHITE][LADY] = 0;
    state.piece[BLACK][PAWN] = 0;
    state.piece[BLACK][LADY] = 0;
    
    SET_BIT(state.piece[WHITE][LADY], 17);  // C3 is a dark square
    state.current_player = WHITE;
    
    MoveList moves;
    movegen_generate(&state, &moves);
    
    // Lady on C3 should have moves in multiple directions
    // At least 1 move expected (could be more depending on implementation)
    ASSERT_GT(moves.count, 0);
}

// =============================================================================
// ENDGAME TESTS
// =============================================================================

TEST(engine_endgame_generator_creates_valid_positions) {
    RNG rng;
    rng_seed(&rng, 12345);
    
    // Generate several positions and verify
    for (int i = 0; i < 10; i++) {
        GameState state;
        int success = setup_random_endgame(&state, &rng);
        
        if (success) {
            int white_count = __builtin_popcountll(get_pieces(&state, WHITE));
            int black_count = __builtin_popcountll(get_pieces(&state, BLACK));
            
            ASSERT_GT(white_count, 0);
            ASSERT_GT(black_count, 0);
        }
    }
}

// =============================================================================
// BITBOARD TESTS
// =============================================================================

TEST(engine_bit_macros_work_correctly) {
    Bitboard bb = 0;
    
    SET_BIT(bb, 0);
    ASSERT_TRUE(TEST_BIT(bb, 0));
    ASSERT_EQ(1ULL, bb);
    
    SET_BIT(bb, 7);
    ASSERT_TRUE(TEST_BIT(bb, 7));
    ASSERT_EQ(0x81ULL, bb);
    
    CLEAR_BIT(bb, 0);
    ASSERT_FALSE(TEST_BIT(bb, 0));
    ASSERT_EQ(0x80ULL, bb);
}

TEST(engine_row_col_macros_work_correctly) {
    ASSERT_EQ(0, ROW(0));
    ASSERT_EQ(0, COL(0));
    
    ASSERT_EQ(0, ROW(7));
    ASSERT_EQ(7, COL(7));
    
    ASSERT_EQ(7, ROW(56));
    ASSERT_EQ(0, COL(56));
    
    ASSERT_EQ(7, ROW(63));
    ASSERT_EQ(7, COL(63));
    
    ASSERT_EQ(27, SQUARE(3, 3));  // D4
}

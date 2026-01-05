/**
 * test_main.c - Unit Test Runner
 * 
 * Compiles and runs all unit tests.
 * 
 * Usage:
 *   ./bin/run_tests           - Run all tests
 *   ./bin/run_tests engine    - Run only engine tests
 *   ./bin/run_tests search    - Run only search tests
 *   ./bin/run_tests neural    - Run only neural tests
 *   ./bin/run_tests training  - Run only training tests
 *   ./bin/run_tests common    - Run only common tests
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Include the test framework first
#include "test_framework.h"

// Include project headers
#include "dama/engine/game.h"
#include "dama/engine/movegen.h"
#include "dama/engine/zobrist.h"
#include "dama/training/endgame.h"
#include "dama/search/mcts.h"
#include "dama/search/mcts_types.h"
#include "dama/neural/cnn.h"
#include "dama/training/dataset.h"
#include "dama/common/rng.h"
#include "dama/common/params.h"
#include "dama/common/error_codes.h"
#include "dama/common/debug.h"

// Include all test files
#include "test_engine.c"
#include "test_search.c"
#include "test_neural.c"
#include "test_training.c"
#include "test_common.c"

// =============================================================================
// TEST REGISTRATION
// =============================================================================

static void register_all_tests(void) {
    // Engine tests
    REGISTER_TEST(engine_init_game_sets_correct_initial_position);
    REGISTER_TEST(engine_get_pieces_returns_all_pieces_for_color);
    REGISTER_TEST(engine_get_all_occupied_returns_all_pieces);
    REGISTER_TEST(engine_check_bit_works_correctly);
    REGISTER_TEST(engine_zobrist_keys_are_unique);
    REGISTER_TEST(engine_hash_changes_after_move);
    REGISTER_TEST(engine_movegen_generate_initial_position_has_7_moves);
    REGISTER_TEST(engine_movegen_generate_simple_only_non_captures);
    REGISTER_TEST(engine_captures_are_mandatory);
    REGISTER_TEST(engine_apply_move_switches_player);
    REGISTER_TEST(engine_apply_move_updates_board);
    REGISTER_TEST(engine_promotion_to_lady);
    REGISTER_TEST(engine_lady_moves_all_directions);
    REGISTER_TEST(engine_endgame_generator_creates_valid_positions);
    REGISTER_TEST(engine_bit_macros_work_correctly);
    REGISTER_TEST(engine_row_col_macros_work_correctly);
    
    // Search tests
    REGISTER_TEST(search_arena_init_and_alloc);
    REGISTER_TEST(search_arena_reset_clears_used);
    REGISTER_TEST(search_arena_alloc_multiple);
    REGISTER_TEST(search_mcts_presets_have_valid_configs);
    REGISTER_TEST(search_create_root_returns_valid_node);
    REGISTER_TEST(search_create_root_generates_children);
    REGISTER_TEST(search_mcts_search_returns_valid_move);
    REGISTER_TEST(search_mcts_search_increases_visits);
    REGISTER_TEST(search_mcts_stats_are_collected);
    REGISTER_TEST(search_get_tree_depth_returns_positive);
    REGISTER_TEST(search_get_tree_node_count_matches_stats);
    REGISTER_TEST(search_mcts_get_policy_sums_to_one);
    REGISTER_TEST(search_mcts_get_policy_nonzero_entries);
    // New search tests
    REGISTER_TEST(search_ucb1_score_increases_with_visits);
    REGISTER_TEST(search_puct_uses_priors);
    REGISTER_TEST(search_backprop_updates_scores);
    REGISTER_TEST(search_virtual_loss_zeroed_after_search);
    REGISTER_TEST(search_solver_detects_terminal);
    REGISTER_TEST(search_tt_create_and_free);
    REGISTER_TEST(search_tt_mask_is_power_of_two);
    REGISTER_TEST(search_tree_reuse_preserves_stats);
    
    // Neural tests
    REGISTER_TEST(neural_cnn_init_allocates_weights);
    REGISTER_TEST(neural_cnn_init_sets_bn_defaults);
    REGISTER_TEST(neural_cnn_free_doesnt_crash);
    REGISTER_TEST(neural_cnn_encode_initial_position);
    REGISTER_TEST(neural_flip_square_works_correctly);
    REGISTER_TEST(neural_cnn_forward_produces_valid_output);
    REGISTER_TEST(neural_cnn_forward_with_history);
    REGISTER_TEST(neural_cnn_forward_is_deterministic);
    REGISTER_TEST(neural_move_to_index_returns_valid_index);
    REGISTER_TEST(neural_move_to_index_different_for_colors);
    REGISTER_TEST(neural_cnn_save_and_load_roundtrip);
    REGISTER_TEST(neural_cnn_load_nonexistent_fails);
    
    // Training tests
    REGISTER_TEST(training_dataset_save_and_load);
    REGISTER_TEST(training_dataset_append);
    REGISTER_TEST(training_dataset_load_alloc);
    REGISTER_TEST(training_dataset_get_count_nonexistent);
    REGISTER_TEST(training_dataset_shuffle_changes_order);
    REGISTER_TEST(training_dataset_split);
    REGISTER_TEST(training_sample_struct_size);
    REGISTER_TEST(training_sample_policy_sums_to_valid);
    REGISTER_TEST(training_cnn_train_step_reduces_loss);
    REGISTER_TEST(training_cnn_gradients_cleared_after_update);
    
    // Common tests
    REGISTER_TEST(common_rng_seed_sets_state);
    REGISTER_TEST(common_rng_seed_zero_becomes_one);
    REGISTER_TEST(common_rng_u32_changes_state);
    REGISTER_TEST(common_rng_u32_is_deterministic);
    REGISTER_TEST(common_rng_u32_different_seeds_differ);
    REGISTER_TEST(common_rng_f32_in_range);
    REGISTER_TEST(common_rng_f32_distribution);
    REGISTER_TEST(common_rng_gamma_positive);
    REGISTER_TEST(common_rng_gamma_alpha_less_than_one);
    REGISTER_TEST(common_rng_gamma_mean_approximately_alpha);
    REGISTER_TEST(common_params_time_limits_positive);
    REGISTER_TEST(common_params_arena_size_reasonable);
    REGISTER_TEST(common_params_cnn_defaults_valid);
    REGISTER_TEST(common_params_mcts_weights_defined);
    REGISTER_TEST(common_bit_operations_correct);
    REGISTER_TEST(common_popcount_works);
    REGISTER_TEST(common_error_codes_defined);
    REGISTER_TEST(common_debug_dbg_not_null_passes);
    REGISTER_TEST(common_debug_dbg_valid_sq_passes);
    REGISTER_TEST(common_debug_dbg_valid_color_passes);
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char **argv) {
    // Seed random for any tests that need it
    srand(time(NULL));
    
    // Initialize game systems needed for tests
    zobrist_init();
    movegen_init();
    
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║           MCTS Dama Unit Test Suite              ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    
    // Register all tests
    register_all_tests();
    
    // Optional filter
    const char *filter = (argc > 1) ? argv[1] : NULL;
    
    if (filter) {
        printf("\nFilter: %s\n", filter);
    }
    
    run_tests(filter);
    
    // Return non-zero if any tests failed
    return g_tests_failed > 0 ? 1 : 0;
}

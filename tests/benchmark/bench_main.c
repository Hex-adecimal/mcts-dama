/**
 * bench_main.c - Benchmark Runner
 * 
 * Runs all performance benchmarks for MCTS Dama.
 * 
 * Usage:
 *   ./bin/run_bench           - Run all benchmarks
 *   ./bin/run_bench engine    - Only engine benchmarks
 *   ./bin/run_bench neural    - Only neural benchmarks
 *   ./bin/run_bench mcts      - Only MCTS benchmarks
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "dama/engine/game.h"
#include "dama/engine/movegen.h"
#include "dama/engine/zobrist.h"
#include "dama/training/endgame.h"
#include "dama/search/mcts.h"
#include "dama/neural/cnn.h"
#include "dama/training/dataset.h"
#include "dama/common/rng.h"
#include "dama/common/params.h"

// =============================================================================
// TIMING UTILITIES
// =============================================================================

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void print_result(const char *name, int iterations, double total_ms) {
    double ops_per_sec = iterations / (total_ms / 1000.0);
    double avg_us = (total_ms * 1000.0) / iterations;
    printf("║ %-35s │ %10.0f │ %12.2f │ %8d ║\n", name, ops_per_sec, avg_us, iterations);
}

static void print_header(void) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                        BENCHMARK RESULTS                               ║\n");
    printf("╠════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ %-35s │ %10s │ %12s │ %8s ║\n", "Benchmark", "Ops/sec", "Avg (μs)", "Iters");
    printf("╠════════════════════════════════════════════════════════════════════════╣\n");
}

static void print_section(const char *name) {
    printf("╠────────────────────────────────────────────────────────────────────────╣\n");
    printf("║ %-70s ║\n", name);
    printf("╠════════════════════════════════════════════════════════════════════════╣\n");
}

static void print_footer(void) {
    printf("╚════════════════════════════════════════════════════════════════════════╝\n\n");
}

#define TARGET_TIME_MS 1000.0
#define MIN_ITERATIONS 10

// =============================================================================
// ENGINE BENCHMARKS
// =============================================================================

static void bench_engine(void) {
    print_section("ENGINE MODULE");
    
    // Move generation - initial position
    {
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            GameState state;
            init_game(&state);
            MoveList moves;
            movegen_generate(&state, &moves);
            iter++;
        }
        print_result("movegen: initial position", iter, get_time_ms() - start);
    }
    
    // Move generation - midgame
    {
        GameState base_state;
        init_game(&base_state);
        MoveList moves;
        for (int i = 0; i < 10; i++) {
            movegen_generate(&base_state, &moves);
            if (moves.count > 0) apply_move(&base_state, &moves.moves[0]);
        }
        
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            movegen_generate(&base_state, &moves);
            iter++;
        }
        print_result("movegen: midgame position", iter, get_time_ms() - start);
    }
    
    // Apply move
    {
        GameState state;
        init_game(&state);
        MoveList moves;
        movegen_generate(&state, &moves);
        Move m = moves.moves[0];
        
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            GameState copy = state;
            apply_move(&copy, &m);
            iter++;
        }
        print_result("apply_move: simple move", iter, get_time_ms() - start);
    }
    
    // Init game + hash
    {
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            GameState state;
            init_game(&state);
            iter++;
        }
        print_result("init_game + zobrist", iter, get_time_ms() - start);
    }
    
    // Explicit Zobrist Compute Hash
    {
        GameState state;
        init_game(&state);
        
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            volatile uint64_t hash = zobrist_compute_hash(&state);
            (void)hash;
            iter++;
        }
        print_result("zobrist_compute_hash", iter, get_time_ms() - start);
    }
    
    // Endgame generation
    {
        RNG rng;
        rng_seed(&rng, 12345);
        
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            GameState state;
            setup_random_endgame(&state, &rng);
            iter++;
        }
        print_result("endgame: random position", iter, get_time_ms() - start);
    }
}

// =============================================================================
// NEURAL NETWORK BENCHMARKS
// =============================================================================

static void bench_neural(void) {
    print_section("NEURAL NETWORK MODULE");
    
    CNNWeights weights;
    cnn_init(&weights);
    
    GameState state;
    init_game(&state);
    
    // Single forward pass
    {
        TrainingSample sample = {0};
        sample.state = state;
        CNNOutput out;
        
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            cnn_forward_sample(&weights, &sample, &out);
            iter++;
        }
        print_result("cnn_forward: single", iter, get_time_ms() - start);
    }
    
    // Forward with history
    {
        GameState hist1 = state;
        CNNOutput out;
        
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            cnn_forward_with_history(&weights, &state, &hist1, NULL, &out);
            iter++;
        }
        print_result("cnn_forward_with_history", iter, get_time_ms() - start);
    }
    
    // Batch forward
    {
        #define BATCH_SIZE 16
        GameState states[BATCH_SIZE];
        CNNOutput outputs[BATCH_SIZE];
        const GameState *state_ptrs[BATCH_SIZE];
        
        for (int i = 0; i < BATCH_SIZE; i++) {
            init_game(&states[i]);
            state_ptrs[i] = &states[i];
        }
        
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            cnn_forward_batch(&weights, state_ptrs, NULL, NULL, outputs, BATCH_SIZE);
            iter++;
        }
        print_result("cnn_forward_batch: 16", iter, get_time_ms() - start);
        #undef BATCH_SIZE
    }
    
    // Encoding
    {
        TrainingSample sample = {0};
        sample.state = state;
        float tensor[CNN_INPUT_CHANNELS * 64];
        float player;
        
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            cnn_encode_sample(&sample, tensor, &player);
            iter++;
        }
        print_result("cnn_encode_sample", iter, get_time_ms() - start);
    }
    
    // Move index conversion
    {
        Move m = {0};
        m.path[0] = 11;
        m.path[1] = 20;
        
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            volatile int idx = cnn_move_to_index(&m, WHITE);
            (void)idx;
            iter++;
        }
        print_result("cnn_move_to_index", iter, get_time_ms() - start);
    }
    
    cnn_free(&weights);
}

// =============================================================================
// MCTS BENCHMARKS
// =============================================================================

static void bench_mcts(void) {
    print_section("MCTS MODULE");
    
    // Root creation
    {
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            GameState state;
            init_game(&state);
            Arena arena;
            arena_init(&arena, ARENA_SIZE_BENCHMARK);
            MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
            Node *root = mcts_create_root(state, &arena, config);
            (void)root;
            arena_free(&arena);
            iter++;
        }
        print_result("mcts_create_root", iter, get_time_ms() - start);
    }
    
    // MCTS 100 nodes Vanilla
    {
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            GameState state;
            init_game(&state);
            Arena arena;
            arena_init(&arena, ARENA_SIZE_BENCHMARK);
            MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
            config.max_nodes = 100;
            Node *root = mcts_create_root(state, &arena, config);
            mcts_search(root, &arena, 10.0, config, NULL, NULL);
            arena_free(&arena);
            iter++;
        }
        print_result("mcts: 100 nodes (Vanilla)", iter, get_time_ms() - start);
    }
    
    // MCTS 500 nodes Vanilla
    {
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            GameState state;
            init_game(&state);
            Arena arena;
            arena_init(&arena, ARENA_SIZE_BENCHMARK);
            MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
            config.max_nodes = 500;
            Node *root = mcts_create_root(state, &arena, config);
            mcts_search(root, &arena, 10.0, config, NULL, NULL);
            arena_free(&arena);
            iter++;
        }
        print_result("mcts: 500 nodes (Vanilla)", iter, get_time_ms() - start);
    }
    
    // MCTS 100 nodes Grandmaster
    {
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            GameState state;
            init_game(&state);
            Arena arena;
            arena_init(&arena, ARENA_SIZE_BENCHMARK);
            MCTSConfig config = mcts_get_preset(MCTS_PRESET_GRANDMASTER);
            config.max_nodes = 100;
            Node *root = mcts_create_root(state, &arena, config);
            mcts_search(root, &arena, 10.0, config, NULL, NULL);
            arena_free(&arena);
            iter++;
        }
        print_result("mcts: 100 nodes (Grandmaster)", iter, get_time_ms() - start);
    }
    
    // Arena allocation
    {
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            Arena arena;
            arena_init(&arena, ARENA_SIZE_BENCHMARK);
            for (int i = 0; i < 1000; i++) {
                arena_alloc(&arena, sizeof(Node));
            }
            arena_free(&arena);
            iter++;
        }
        print_result("arena_alloc: 1000 nodes", iter, get_time_ms() - start);
    }
    
    // MCTS 1000 nodes with AlphaZero config (heavier)
    {
        CNNWeights weights;
        cnn_init(&weights);
        
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            GameState state;
            init_game(&state);
            Arena arena;
            arena_init(&arena, ARENA_SIZE_BENCHMARK);
            MCTSConfig config = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO);
            config.max_nodes = 1000;
            config.cnn_weights = &weights;
            Node *root = mcts_create_root(state, &arena, config);
            mcts_search(root, &arena, 10.0, config, NULL, NULL);
            arena_free(&arena);
            iter++;
        }
        print_result("mcts: 1000 nodes (AlphaZero+CNN)", iter, get_time_ms() - start);
        
        cnn_free(&weights);
    }
    
    // Transposition table operations
    {
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            TranspositionTable *tt = tt_create(4096);
            if (!tt) continue;
            
            Arena arena;
            arena_init(&arena, ARENA_SIZE_BENCHMARK);
            MCTSConfig config = mcts_get_preset(MCTS_PRESET_VANILLA);
            
            // Just benchmark TT create/free since insert/lookup need more setup
            arena_free(&arena);
            tt_free(tt);
            iter++;
        }
        print_result("tt: create+free (4096)", iter, get_time_ms() - start);
    }
}


// =============================================================================
// TRAINING BENCHMARKS
// =============================================================================

static void bench_training(void) {
    print_section("TRAINING MODULE");
    
    CNNWeights weights;
    cnn_init(&weights);
    
    // Single train step
    {
        TrainingSample sample = {0};
        init_game(&sample.state);
        sample.target_policy[0] = 1.0f;
        sample.target_value = 0.5f;
        float p, v;
        
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            cnn_train_step(&weights, &sample, 1, 0.01f, 0.001f, 0.0f, 0.0f, &p, &v);
            iter++;
        }
        print_result("cnn_train_step: batch=1", iter, get_time_ms() - start);
    }
    
    // Batch train step
    {
        #define BATCH 32
        TrainingSample samples[BATCH];
        memset(samples, 0, sizeof(samples));
        for (int i = 0; i < BATCH; i++) {
            init_game(&samples[i].state);
            samples[i].target_policy[i % CNN_POLICY_SIZE] = 1.0f;
            samples[i].target_value = 0.0f;
        }
        float p, v;
        
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            cnn_train_step(&weights, samples, BATCH, 0.01f, 0.001f, 0.0f, 0.0f, &p, &v);
            iter++;
        }
        print_result("cnn_train_step: batch=32", iter, get_time_ms() - start);
        #undef BATCH
    }
    
    cnn_free(&weights);
    
    // Dataset shuffle
    {
        #define N_SAMPLES 1000
        TrainingSample *samples = malloc(N_SAMPLES * sizeof(TrainingSample));
        memset(samples, 0, N_SAMPLES * sizeof(TrainingSample));
        
        int iter = 0;
        double start = get_time_ms();
        while (get_time_ms() - start < TARGET_TIME_MS || iter < MIN_ITERATIONS) {
            dataset_shuffle(samples, N_SAMPLES);
            iter++;
        }
        print_result("dataset_shuffle: 1000", iter, get_time_ms() - start);
        
        free(samples);
        #undef N_SAMPLES
    }
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char **argv) {
    // Initialize
    srand(time(NULL));
    zobrist_init();
    movegen_init();
    
    const char *filter = (argc > 1) ? argv[1] : NULL;
    
    print_header();
    
    if (!filter || strcmp(filter, "engine") == 0) {
        bench_engine();
    }
    
    if (!filter || strcmp(filter, "neural") == 0) {
        bench_neural();
    }
    
    if (!filter || strcmp(filter, "mcts") == 0) {
        bench_mcts();
    }
    
    if (!filter || strcmp(filter, "training") == 0) {
        bench_training();
    }
    
    print_footer();
    
    return 0;
}

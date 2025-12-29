/**
 * cmd_train.c - Neural Network Training Command (Controller)
 * 
 * Orchestrates self-play data generation and model training.
 * Usage: dama train [options]
 */

#include "logging.h"
#include "../../src/ui/cli_view.h"
#include "../../src/nn/selfplay.h"
#include "../../src/nn/training_pipeline.h"
#include "../../src/core/movegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// CALLBACKS (UI)
// =============================================================================

// --- Selfplay Callbacks ---

static void sp_on_start(int total) {
    log_printf("\nStarting self-play generation (%d games)...\n", total);
}

static void sp_on_progress(int completed, int total, int w, int l, int d) {
    // Interactive progress bar
    printf("\rGenerated: %d/%d | W:%d L:%d D:%d ", completed, total, w, l, d);
    fflush(stdout);
}

static void sp_on_game_complete(int g, int total, int res, int moves, int reason) {
    (void)g; (void)total; (void)res; (void)moves; (void)reason;
    // Optional: Log every game to debug file?
}

// --- Training Callbacks ---

static void tr_on_init(int total, int val) {
    log_printf("\nLoaded dataset: %s training samples, %s validation.\n", 
               format_num(total), format_num(val));
}

static void tr_on_epoch_start(int epoch, int total, float lr) {
    log_printf("\nEpoch %d/%d (LR: %.6f)\n", epoch, total, lr);
    log_printf("----------------------------------------------------------------\n");
}

static void tr_on_batch(int b, int total, float loss, int samples) {
    if (b % 10 == 0 || b == total - 1) {
        printf("\rBatch %d/%d | Loss: %.4f | Samples: %d", b, total, loss, samples);
        fflush(stdout);
    }
}

static void tr_on_epoch_end(int epoch, float avg_loss, float val_loss, float val_acc, int improved, const char *path) {
    (void)epoch;
    printf("\n"); // Clear progress line
    log_printf("Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.1f%%\n", 
               avg_loss, val_loss, val_acc * 100.0f);
    
    if (improved) {
        log_printf(">> New best model saved to %s\n", path ? path : "buffer");
    } else {
        log_printf(".. No improvement\n");
    }
}

static void tr_on_complete(float best, int epoch) {
    TrainingResultView view = {
        .weights_file = "checkpoints", // Placeholder or pass path
        .best_loss = best,
        .training_time = 0, 
        .samples_per_sec = 0
    };
    cli_view_print_training_complete(&view);
}

// =============================================================================
// CONFIGURATION
// =============================================================================

static int get_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// =============================================================================
// CMD_TRAIN
// =============================================================================

int cmd_train(int argc, char **argv) {
    // Defaults
    SelfplayConfig sp_cfg = {
        .games = 1000,
        .max_moves = 200,
        .time_limit = 0.5, // Not used much if using nodes
        .temp = 1.0f,
        .output_file = "out/data/selfplay.dat",
        .parallel_threads = get_max_threads(),
        .mercy = { .threshold = 3.0f, .check_interval = 10 },
        .overwrite_data = 0,
        .on_start = sp_on_start,
        .on_progress = sp_on_progress,
        .on_game_complete = sp_on_game_complete
    };
    
    TrainingPipelineConfig tr_cfg = {
        .epochs = 10,
        .batch_size = 64,
        .learning_rate = 0.0005f,
        .l2_decay = 1e-4f,
        .momentum = 0.9f,
        .patience = 5,
        .data_path = "out/data/selfplay.dat",
        .model_path = NULL, // Will determine
        .num_threads = get_max_threads(),
        .on_init = tr_on_init,
        .on_epoch_start = tr_on_epoch_start,
        .on_batch_log = tr_on_batch,
        .on_epoch_end = tr_on_epoch_end,
        .on_complete = tr_on_complete
    };
    
    int selfplay_only = 0;
    int train_only = 0;
    int fresh_init = 0;
    const char *weights_in = "out/models/cnn_weights_v3.bin";
    const char *weights_out = "out/models/cnn_weights_v3.bin";
    
    // Parse Args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--selfplay") == 0) selfplay_only = 1;
        else if (strcmp(argv[i], "--train") == 0) train_only = 1;
        else if (strcmp(argv[i], "--games") == 0 && i+1 < argc) sp_cfg.games = atoi(argv[++i]);
        else if (strcmp(argv[i], "--epochs") == 0 && i+1 < argc) tr_cfg.epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) tr_cfg.learning_rate = atof(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i+1 < argc) tr_cfg.batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--l2") == 0 && i+1 < argc) tr_cfg.l2_decay = atof(argv[++i]);
        else if (strcmp(argv[i], "--patience") == 0 && i+1 < argc) tr_cfg.patience = atoi(argv[++i]);
        else if (strcmp(argv[i], "--data") == 0 && i+1 < argc) {
            sp_cfg.output_file = argv[++i];
            tr_cfg.data_path = sp_cfg.output_file;
        }
        else if (strcmp(argv[i], "--weights") == 0 && i+1 < argc) {
            weights_in = argv[++i];
            weights_out = argv[i];
        }
        else if (strcmp(argv[i], "--overwrite") == 0) sp_cfg.overwrite_data = 1;
        else if (strcmp(argv[i], "--init") == 0) fresh_init = 1;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: dama train [options]\n");
            // ... print help ...
            return 0;
        }
    }
    
    tr_cfg.model_path = weights_out;
    tr_cfg.backup_path = "out/models/cnn_weights_backup.bin";
    
    // Setup
    zobrist_init();
    init_move_tables();
    
    // Load Model
    CNNWeights weights;
    cnn_init(&weights);
    if (!fresh_init && cnn_load_weights(&weights, weights_in) == 0) {
        log_printf("Loaded weights from %s\n", weights_in);
    } else {
        log_printf("Initialized fresh weights\n");
    }
    
    // MCTS Config for Selfplay
    MCTSConfig mcts_cfg = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO);
    mcts_cfg.cnn_weights = &weights;
    mcts_cfg.max_nodes = 800; // Default Selfplay nodes
    
    // 1. Selfplay
    if (!train_only) {
        // UI Header
        SelfplayView sp_view = {
            .output_file = sp_cfg.output_file,
            .num_games = sp_cfg.games,
            .omp_threads = sp_cfg.parallel_threads,
            .mcts_nodes = mcts_cfg.max_nodes
        };
        cli_view_print_selfplay(&sp_view); // Print header
        
        selfplay_run(&sp_cfg, &mcts_cfg);
        
        log_printf("\nSelf-play complete.\n");
    }
    
    // 2. Training
    if (!selfplay_only) {
        TrainingConfigView tr_view = {
            .epochs = tr_cfg.epochs,
            .batch_size = tr_cfg.batch_size,
            .learning_rate = tr_cfg.learning_rate,
            .total_params = 0
        };
        cli_view_print_training_config(&tr_view);
        
        training_run(&weights, &tr_cfg);
    }
    
    cnn_free(&weights);
    return 0;
}
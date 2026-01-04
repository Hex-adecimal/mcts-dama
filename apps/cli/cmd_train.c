/**
 * cmd_train.c - Neural Network Training Command (Controller)
 * 
 * Orchestrates self-play data generation and model training.
 * Usage: dama train [options]
 */

#include "dama/common/logging.h"
#include "dama/common/cli_view.h"
#include "dama/training/selfplay.h"
#include "dama/training/training_pipeline.h"
#include "dama/engine/movegen.h"
#include "dama/common/params.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

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
    const char* winners[] = {"Draw ", "White", "Black"};
    const char* reasons[] = {"Normal", "Resign", "Mercy ", "Stale ", "MaxMov", "Repet "};
    printf("\n  > Game %3d/%3d | Result: %s | Moves: %3d | Case: %s", 
           g + 1, total, winners[res], moves, reasons[reason]);
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

static int training_val_interval = 10;

static void tr_on_batch(int b, int total, float p_loss, float v_loss, int samples) {
    if (b % training_val_interval == 0 || b == total - 1) {
        printf("\rBatch %d/%d | P:%.4f V:%.4f | Samples: %d", b, total, p_loss, v_loss, samples);
        fflush(stdout);
    }
}

static void tr_on_epoch_end(int epoch, float train_p, float train_v, float val_p, float val_v, float val_acc, int improved, const char *path) {
    (void)epoch;
    printf("\n"); // Clear progress line
    log_printf("Train | Policy: %.4f  Value: %.4f  Total: %.4f\n", train_p, train_v, train_p + train_v);
    log_printf("Valid | Policy: %.4f  Value: %.4f  Total: %.4f | Acc: %.1f%%\n", 
               val_p, val_v, val_p + val_v, val_acc * 100.0f);
    
    if (improved) {
        log_printf(">> New best model saved to %s\n", path ? path : "buffer");
    } else {
        log_printf(".. No improvement\n");
    }
}

// Static storage for training metrics
static time_t training_start_time = 0;
static const char *training_model_path = NULL;
static int training_total_samples = 0;

static void tr_on_complete(float best, int epoch) {
    (void)epoch;
    double elapsed = difftime(time(NULL), training_start_time);
    double sps = (elapsed > 0) ? (double)training_total_samples / elapsed : 0;
    
    TrainingResultView view = {
        .weights_file = training_model_path ? training_model_path : "N/A",
        .best_loss = best,
        .training_time = elapsed, 
        .samples_per_sec = sps
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
        .output_file = "out/data/active.dat",
        .parallel_threads = get_max_threads(),
        .overwrite_data = 0,
        .on_start = sp_on_start,
        .on_progress = sp_on_progress,
        .on_game_complete = sp_on_game_complete,
        .endgame_prob = 0.0f
    };
    
    TrainingPipelineConfig tr_cfg = {
        .epochs = CNN_DEFAULT_EPOCHS,
        .batch_size = CNN_DEFAULT_BATCH_SIZE,
        .learning_rate = 0.0f,  // Not used anymore - LRs are set in training_pipeline.c
        .l2_decay = CNN_DEFAULT_L2_DECAY,
        .momentum = CNN_DEFAULT_MOMENTUM,
        .patience = CNN_PATIENCE,
        .data_path = "out/data/active.dat",
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
    const char *weights_in = "out/models/cnn_weights.bin";
    const char *weights_out = "out/models/cnn_weights.bin";
    
    // Override parameters (0 = use defaults)
    int nodes_override = 0;
    int threads_override = 0;
    
    // Parse Args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--selfplay") == 0) selfplay_only = 1;
        else if (strcmp(argv[i], "--train") == 0) train_only = 1;
        else if (strcmp(argv[i], "--games") == 0 && i+1 < argc) sp_cfg.games = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i+1 < argc) sp_cfg.temp = atof(argv[++i]);
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
        else if (strcmp(argv[i], "--nodes") == 0 && i+1 < argc) nodes_override = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i+1 < argc) threads_override = atoi(argv[++i]);
        else if (strcmp(argv[i], "--endgame-prob") == 0 && i+1 < argc) sp_cfg.endgame_prob = atof(argv[++i]);
        else if (strcmp(argv[i], "--val-interval") == 0 && i+1 < argc) training_val_interval = atoi(argv[++i]);
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
    
    // Apply command-line overrides
    if (nodes_override > 0) {
        mcts_cfg.max_nodes = nodes_override;
    }
    if (threads_override > 0) {
        sp_cfg.parallel_threads = threads_override;
        tr_cfg.num_threads = threads_override;
    }
    
    // 1. Selfplay
    if (!train_only) {
        // Prepare Date String
        time_t now = time(NULL);
        char date_str[64];
        strftime(date_str, sizeof(date_str), "%Y-%m-%d %H:%M:%S", localtime(&now));

        // UI Header
        SelfplayView sp_view = {
            .output_file = sp_cfg.output_file,
            .num_games = sp_cfg.games,
            .omp_threads = sp_cfg.parallel_threads,
            .mcts_nodes = mcts_cfg.max_nodes,
            .date_str = date_str,
            // Dirichlet
            .dirichlet_alpha = 0.3f, // Dama/Checkers value (0.3 is standard for Chess, Dama similiar branching?)
            .dirichlet_epsilon = 0.25f,
            // Temp
            .temperature = sp_cfg.temp,
            .temp_threshold = 30, // Default in selfplay.c usually
            .max_moves = sp_cfg.max_moves,
            .endgame_prob = sp_cfg.endgame_prob
        };
        cli_view_print_selfplay(&sp_view); // Print header
        
        selfplay_run(&sp_cfg, &mcts_cfg);
        
        log_printf("\nSelf-play complete.\n");
    }
    
    // 2. Training
    if (!selfplay_only) {
        // Compute total parameters: conv layers + BN + FC heads
        // Conv1: 64*12*9=6912, Conv2-4: 64*64*9=36864 each, BN: 64*8, Policy: 512*4097, Value: 256*4097+256+1
        int total_params = 6912 + 3*36864 + 64*8 + 512*4097 + 256*4097 + 256 + 1;
        
        TrainingConfigView tr_view = {
            .epochs = tr_cfg.epochs,
            .batch_size = tr_cfg.batch_size,
            // Calculate effective base LR to show reality
            .learning_rate = (tr_cfg.learning_rate > 1e-6f) ? tr_cfg.learning_rate : CNN_POLICY_LR,
            .l2_decay = tr_cfg.l2_decay,
            .patience = tr_cfg.patience,
            .omp_threads = tr_cfg.num_threads,
            .total_params = total_params
        };
        cli_view_print_training_config(&tr_view);
        
        // Initialize tracking for completion message
        training_start_time = time(NULL);
        training_model_path = tr_cfg.model_path;
        training_total_samples = dataset_get_count(tr_cfg.data_path) * tr_cfg.epochs;
        
        training_run(&weights, &tr_cfg);
    }
    
    cnn_free(&weights);
    return 0;
}
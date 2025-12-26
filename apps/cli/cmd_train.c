/**
 * cmd_train.c - Training command implementation
 * Handles: selfplay data generation + CNN training
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <locale.h>
#include <unistd.h>
#include <sys/stat.h>
#include <omp.h>

#include "cmd_data.h"
#include "game.h"
#include "movegen.h"
#include "cnn.h"
#include "mcts.h"
#include "dataset.h"
#include "params.h"
#include "mcts_presets.h"

// =============================================================================
// DEFAULT CONFIGURATION
// =============================================================================

// Training
#define DEFAULT_WEIGHTS_FILE    "models/cnn_weights.bin"
#define DEFAULT_TRAIN_FILE      "data/selfplay_games.bin"
#define DEFAULT_BATCH_SIZE      64
#define DEFAULT_EPOCHS          100
#define DEFAULT_LR              0.0005f
#define DEFAULT_L2_DECAY        1e-4f
#define DEFAULT_PATIENCE        3

// Selfplay
#define DEFAULT_SELFPLAY_NODES          800
#define DEFAULT_TEMP_THRESHOLD          60
#define DEFAULT_DIRICHLET_ALPHA         0.3f
#define DEFAULT_DIRICHLET_EPSILON       0.25f
#define DEFAULT_MAX_MOVES               200

// =============================================================================
// THREAD-SAFE RNG (xorshift32)
// =============================================================================

typedef struct { uint32_t state; } RNG;

static inline void rng_seed(RNG *r, uint32_t s) { r->state = s ? s : 1; }

static inline uint32_t rng_u32(RNG *r) {
    uint32_t x = r->state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return r->state = x;
}

static inline float rng_f32(RNG *r) { return (float)rng_u32(r) / (float)UINT32_MAX; }

static float rng_gamma(RNG *r, float alpha) {
    if (alpha < 1.0f) return rng_gamma(r, 1.0f + alpha) * powf(rng_f32(r), 1.0f / alpha);
    float d = alpha - 1.0f/3.0f, c = 1.0f / sqrtf(9.0f * d);
    while (1) {
        float x, v;
        do { x = sqrtf(-2.0f * logf(rng_f32(r))) * cosf(2.0f * M_PI * rng_f32(r)); v = 1.0f + c * x; } while (v <= 0);
        v = v * v * v;
        float u = rng_f32(r);
        if (u < 1.0f - 0.0331f * x*x*x*x || logf(u) < 0.5f * x*x + d * (1.0f - v + logf(v))) return d * v;
    }
}

// =============================================================================
// SELFPLAY HELPERS
// =============================================================================

typedef struct {
    float policy[CNN_POLICY_SIZE];
    GameState state;
    GameState history[2];
    int player;
} GameStep;

static void add_dirichlet_noise(Node *root, RNG *rng, float eps, float alpha) {
    if (!root || root->num_children == 0) return;
    int n = root->num_children;
    float *noise = malloc(n * sizeof(float)), sum = 0;
    for (int i = 0; i < n; i++) { noise[i] = rng_gamma(rng, alpha); sum += noise[i]; }
    for (int i = 0; i < n; i++) root->children[i]->prior = (1-eps) * root->children[i]->prior + eps * noise[i]/sum;
    free(noise);
}

static int game_over(const GameState *s) { MoveList m; generate_moves(s, &m); return m.count == 0; }

static int mercy_rule(const GameState *s, int moves) {
    if (moves < 60) return 0;
    int w = __builtin_popcountll(s->white_pieces | s->white_ladies);
    int b = __builtin_popcountll(s->black_pieces | s->black_ladies);
    int wl = __builtin_popcountll(s->white_ladies), bl = __builtin_popcountll(s->black_ladies);
    if (w + wl >= b + bl + 4 && b <= 3) return 1;
    if (b + bl >= w + wl + 4 && w <= 3) return -1;
    return 0;
}

// Play one self-play game, returns result (+1 white, -1 black, 0 draw)
static int play_selfplay_game(int game_id, const CNNWeights *weights, MCTSConfig cfg_cnn, 
                               MCTSConfig cfg_opp, GameStep *history, int *out_len, RNG *rng) {
    GameState state;
    init_game(&state);
    Arena arena;
    arena_init(&arena, 100 * 1024 * 1024);
    
    int moves = 0;
    float policy[CNN_POLICY_SIZE];
    int hybrid = (game_id % 2 == 0);  // 50% pure CNN, 50% CNN vs opponent
    
    while (!game_over(&state) && moves < DEFAULT_MAX_MOVES) {
        float temp = (moves < DEFAULT_TEMP_THRESHOLD) ? 1.0f : 0.05f;
        
        MCTSConfig cfg;
        int is_cnn = 1;
        if (!hybrid) {
            cfg = cfg_cnn;
        } else {
            int cnn_white = ((game_id / 2) % 2 == 0);
            is_cnn = (state.current_player == WHITE) == cnn_white;
            cfg = is_cnn ? cfg_cnn : cfg_opp;
        }
        
        Node *root = mcts_create_root(state, &arena, cfg);
        
        // Add noise early game for CNN
        if (moves < 30 && is_cnn) {
            mcts_search(root, &arena, 0.0, cfg, NULL, NULL);
            add_dirichlet_noise(root, rng, DEFAULT_DIRICHLET_EPSILON, DEFAULT_DIRICHLET_ALPHA);
        }
        
        mcts_search(root, &arena, 0.0, cfg, NULL, NULL);
        mcts_get_policy(root, policy, temp, &state);
        
        // Select move
        Move chosen = {0};
        if (temp < 0.1f) {
            int best_i = 0;
            for (int i = 1; i < CNN_POLICY_SIZE; i++) if (policy[i] > policy[best_i]) best_i = i;
            for (int i = 0; i < root->num_children; i++) {
                if (cnn_move_to_index(&root->children[i]->move_from_parent, state.current_player) == best_i) {
                    chosen = root->children[i]->move_from_parent;
                    break;
                }
            }
        } else {
            double sum = 0, exp = 1.0 / temp;
            for (int i = 0; i < root->num_children; i++) sum += pow(root->children[i]->visits, exp);
            double sample = rng_f32(rng) * sum, cur = 0;
            for (int i = 0; i < root->num_children; i++) {
                cur += pow(root->children[i]->visits, exp);
                if (cur >= sample) { chosen = root->children[i]->move_from_parent; break; }
            }
            if (chosen.path[0] == 0 && chosen.path[1] == 0 && root->num_children > 0)
                chosen = root->children[0]->move_from_parent;
        }
        
        // Record step
        history[moves].state = state;
        history[moves].player = state.current_player;
        memcpy(history[moves].policy, policy, sizeof(policy));
        if (moves >= 1) history[moves].history[0] = history[moves-1].state;
        if (moves >= 2) history[moves].history[1] = history[moves-2].state;
        
        apply_move(&state, &chosen);
        arena_reset(&arena);
        moves++;
        
        if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) break;
        if (mercy_rule(&state, moves) != 0) break;
        
        // Early resignation check
        if (moves > 40) {
            CNNOutput out;
            cnn_forward_with_history(weights, &state, 
                moves >= 1 ? &history[moves-1].state : NULL,
                moves >= 2 ? &history[moves-2].state : NULL, &out);
            if (out.value < -0.85f) break;
        }
    }
    
    arena_free(&arena);
    *out_len = moves;
    
    int m = mercy_rule(&state, moves);
    if (m != 0) return m;
    if (moves >= DEFAULT_MAX_MOVES || state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) return 0;
    return (state.current_player == WHITE) ? -1 : 1;
}

// =============================================================================
// SELFPLAY MAIN LOOP
// =============================================================================

static int run_selfplay(int num_games, const char *output_file, const char *weights_file) {
    printf("=== Self-Play Data Generation ===\n");
    printf("Games: %d | Output: %s\n\n", num_games, output_file);
    
    zobrist_init();
    init_move_tables();
    
    CNNWeights weights;
    cnn_init(&weights);
    if (cnn_load_weights(&weights, weights_file) != 0) {
        printf("Warning: No weights at %s, using random init\n", weights_file);
    }
    
    MCTSConfig cfg_cnn = { .ucb1_c = 1.5, .use_puct = 1, .puct_c = PUCT_C, 
                           .cnn_weights = (void*)&weights, .max_nodes = DEFAULT_SELFPLAY_NODES };
    
    MCTSConfig opponents[] = {
        mcts_get_preset(MCTS_PRESET_VANILLA),
        mcts_get_preset(MCTS_PRESET_PURE_VANILLA),
        mcts_get_preset(MCTS_PRESET_SMART_ROLLOUTS),
        mcts_get_preset(MCTS_PRESET_SOLVER_ONLY),
    };
    int num_opps = sizeof(opponents) / sizeof(opponents[0]);
    for (int i = 0; i < num_opps; i++) opponents[i].max_nodes = DEFAULT_SELFPLAY_NODES * 2;
    
    remove(output_file);
    
    RNG rng;
    rng_seed(&rng, time(NULL));
    
    int batch_cap = 2000;
    TrainingSample *batch = malloc(batch_cap * sizeof(TrainingSample));
    int batch_count = 0, total_samples = 0;
    
    for (int g = 0; g < num_games; g++) {
        GameStep *history = malloc(DEFAULT_MAX_MOVES * sizeof(GameStep));
        int steps = 0;
        
        MCTSConfig opp = opponents[g % num_opps];
        int result = play_selfplay_game(g, &weights, cfg_cnn, opp, history, &steps, &rng);
        total_samples += steps;
        
        // Convert to training samples
        for (int k = 0; k < steps; k++) {
            if (batch_count >= batch_cap) {
                dataset_save_append(output_file, batch, batch_count);
                batch_count = 0;
            }
            TrainingSample *s = &batch[batch_count++];
            s->state = history[k].state;
            s->history[0] = history[k].history[0];
            s->history[1] = history[k].history[1];
            memcpy(s->target_policy, history[k].policy, sizeof(s->target_policy));
            int sign = (history[k].player == WHITE) ? 1 : -1;
            s->target_value = (result == 0) ? 0.0f : (float)(result * sign);
        }
        
        free(history);
        
        if (g % 10 == 0) {
            printf("\rGames: %d/%d | Samples: %d", g, num_games, total_samples);
            fflush(stdout);
        }
    }
    
    if (batch_count > 0) dataset_save_append(output_file, batch, batch_count);
    free(batch);
    cnn_free(&weights);
    
    printf("\n\nDone! Generated %d samples from %d games.\n", total_samples, num_games);
    return 0;
}

// =============================================================================
// TRAINING CONFIG & HELPERS
// =============================================================================

typedef struct {
    const char *weights_file;
    const char *train_file;
    const char *val_file;
    int epochs;
    int batch_size;
    float learning_rate;
    float l2_decay;
    int patience;
    int init_fresh;
} TrainConfig;

static TrainConfig default_config(void) {
    return (TrainConfig){
        .weights_file = DEFAULT_WEIGHTS_FILE,
        .train_file = DEFAULT_TRAIN_FILE,
        .val_file = NULL,
        .epochs = DEFAULT_EPOCHS,
        .batch_size = DEFAULT_BATCH_SIZE,
        .learning_rate = DEFAULT_LR,
        .l2_decay = DEFAULT_L2_DECAY,
        .patience = DEFAULT_PATIENCE,
        .init_fresh = 0
    };
}

typedef struct { float total, policy, value; } LossMetrics;

static LossMetrics run_validation(CNNWeights *w, TrainingSample *data,
                                   BalancedIndex *idx, size_t count,
                                   TrainingSample *batch, int batch_size) {
    LossMetrics m = {0};
    if (count == 0) return m;
    
    int num_batches = (count / batch_size > 10) ? count / batch_size : 10;
    
    for (int b = 0; b < num_batches; b++) {
        fill_balanced_batch(batch, batch_size, data, idx, count);
        CNNOutput out;
        float bp = 0, bv = 0;
        
        #pragma omp parallel for reduction(+:bp, bv) private(out)
        for (int j = 0; j < batch_size; j++) {
            cnn_forward_sample(w, &batch[j], &out);
            float lp = 0;
            for (int k = 0; k < CNN_POLICY_SIZE; k++) {
                if (batch[j].target_policy[k] > 0) 
                    lp -= batch[j].target_policy[k] * logf(fmaxf(out.policy[k], 1e-4f));
            }
            bp += lp;
            float diff = out.value - batch[j].target_value;
            bv += diff * diff;
        }
        m.policy += bp / batch_size;
        m.value += bv / batch_size;
    }
    
    m.total = (m.policy + m.value) / num_batches;
    m.policy /= num_batches;
    m.value /= num_batches;
    return m;
}

// =============================================================================
// TRAINING LOOP
// =============================================================================

static int run_training(TrainConfig *cfg) {
    printf("=== CNN Training ===\n\n");
    
    srand((unsigned)time(NULL));
    zobrist_init();
    init_move_tables();
    
    printf("Loading: %s\n", cfg->train_file);
    int total = 0;
    TrainingSample *all_data = load_dataset_file(cfg->train_file, &total);
    if (!all_data) return 1;
    
    DatasetSplit split = split_dataset(all_data, total, cfg->val_file, 0.90f);
    
    BalancedIndex train_idx = build_balanced_index(split.train_data, split.train_count);
    BalancedIndex val_idx = build_balanced_index(split.val_data, split.val_count);
    
    CNNWeights weights;
    cnn_init(&weights);
    if (!cfg->init_fresh && cnn_load_weights(&weights, cfg->weights_file) == 0)
        printf("✓ Loaded weights from %s\n", cfg->weights_file);
    else
        printf("⚠ Starting fresh\n");
    
    printf("\nTraining: %d epochs, LR=%.4f, Batch=%d\n\n", cfg->epochs, cfg->learning_rate, cfg->batch_size);
    printf("+-------+---------+---------+---------+------------+\n");
    printf("| Epoch |  Train  |   Val   |   LR    |   Status   |\n");
    printf("+-------+---------+---------+---------+------------+\n");
    
    TrainingSample *batch = malloc(cfg->batch_size * sizeof(TrainingSample));
    float lr = cfg->learning_rate, best = 1e9f;
    int patience = 0;
    
    for (int epoch = 1; epoch <= cfg->epochs; epoch++) {
        float et = 0;
        int nb = split.train_count / cfg->batch_size;
        
        for (int b = 0; b < nb; b++) {
            fill_balanced_batch(batch, cfg->batch_size, split.train_data, &train_idx, split.train_count);
            float eff_lr = (epoch == 1) ? lr * (b+1.0f) / nb : lr;
            float p, v;
            et += cnn_train_step(&weights, batch, cfg->batch_size, eff_lr, 0, cfg->l2_decay, &p, &v);
        }
        et /= nb;
        
        LossMetrics val = run_validation(&weights, split.val_data, &val_idx, split.val_count, batch, cfg->batch_size);
        
        if (val.total < best) {
            best = val.total;
            cnn_save_weights(&weights, cfg->weights_file);
            patience = 0;
            printf("|  %3d  | %7.4f | %7.4f | %.5f |   *BEST*   |\n", epoch, et, val.total, lr);
        } else {
            patience++;
            printf("|  %3d  | %7.4f | %7.4f | %.5f |  wait %d/%d |\n", epoch, et, val.total, lr, patience, cfg->patience);
        }
        
        if (patience >= cfg->patience) {
            lr *= 0.1f;
            if (lr < 1e-6f) { printf("\nConverged!\n"); break; }
            printf("*** LR -> %.6f ***\n", lr);
            patience = 0;
        }
    }
    
    printf("+-------+---------+---------+---------+------------+\n");
    printf("Best: %.4f | Saved: %s\n", best, cfg->weights_file);
    
    free(all_data); free(batch);
    free_balanced_index(&train_idx);
    free_balanced_index(&val_idx);
    if (cfg->val_file && split.val_data) free(split.val_data);
    cnn_training_cleanup();
    cnn_free(&weights);
    
    return 0;
}

// =============================================================================
// COMMAND ENTRY POINT
// =============================================================================

int cmd_train(int argc, char **argv) {
    setlocale(LC_NUMERIC, "");
    
    TrainConfig cfg = default_config();
    int do_selfplay = 0, selfplay_games = 1000;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: dama train [options]\n\n");
            printf("Self-play:\n");
            printf("  --selfplay        Generate data before training\n");
            printf("  --games N         Self-play games (default: 1000)\n");
            printf("\nTraining:\n");
            printf("  --epochs N        Training epochs (default: %d)\n", DEFAULT_EPOCHS);
            printf("  --lr RATE         Learning rate (default: %.4f)\n", DEFAULT_LR);
            printf("  --batch N         Batch size (default: %d)\n", DEFAULT_BATCH_SIZE);
            printf("  --data FILE       Training data file\n");
            printf("  --weights FILE    Weights file\n");
            printf("  --init            Fresh weights\n");
            return 0;
        }
        else if (strcmp(argv[i], "--selfplay") == 0) do_selfplay = 1;
        else if (strcmp(argv[i], "--games") == 0 && i+1 < argc) selfplay_games = atoi(argv[++i]);
        else if (strcmp(argv[i], "--epochs") == 0 && i+1 < argc) cfg.epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) cfg.learning_rate = atof(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i+1 < argc) cfg.batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--data") == 0 && i+1 < argc) cfg.train_file = argv[++i];
        else if (strcmp(argv[i], "--weights") == 0 && i+1 < argc) cfg.weights_file = argv[++i];
        else if (strcmp(argv[i], "--init") == 0) cfg.init_fresh = 1;
    }
    
    if (do_selfplay) {
        if (run_selfplay(selfplay_games, cfg.train_file, cfg.weights_file) != 0)
            return 1;
    }
    
    return run_training(&cfg);
}
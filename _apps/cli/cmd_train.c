/**
 * cmd_train.c - Training command implementation
 * Handles: selfplay data generation + CNN training
 * 
 * NOTE: Depends on cmd_data.c being included first (for DatasetSplit, BalancedIndex)
 * 
 * Structure:
 *   1. Constants & Config
 *   2. RNG utilities
 *   3. Selfplay helpers
 *   4. Training helpers
 *   5. Main functions (run_selfplay, run_training, cmd_train)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <locale.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <omp.h>

#include "game.h"
#include "movegen.h"
#include "cnn.h"
#include "mcts.h"
#include "dataset.h"
#include "params.h"

// =============================================================================
// CONSTANTS
// =============================================================================

// Training defaults
#define DEFAULT_WEIGHTS_FILE    "out/models/cnn_weights.bin"
#define DEFAULT_TRAIN_FILE      "out/data/selfplay_games.bin"
#define DEFAULT_BATCH_SIZE      64
#define DEFAULT_EPOCHS          10
#define DEFAULT_LR              0.0005f
#define DEFAULT_L2_DECAY        1e-4f
#define DEFAULT_PATIENCE        3

// Selfplay defaults
#define DEFAULT_SELFPLAY_NODES          1600
#define DEFAULT_TEMP_THRESHOLD          60
#define DEFAULT_DIRICHLET_ALPHA         0.3f
#define DEFAULT_DIRICHLET_EPSILON       0.25f
#define DEFAULT_MAX_MOVES               250
#define DEFAULT_LOG_DIR                 "out/logs"

// Global log file (dual output: stdout + file)
static FILE *g_logfile = NULL;

// Dual printf: writes to both stdout and log file
static void log_printf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    
    if (g_logfile) {
        va_start(args, fmt);
        vfprintf(g_logfile, fmt, args);
        va_end(args);
        fflush(g_logfile);
    }
}

// =============================================================================
// CONFIGURATION STRUCT
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
    float puct_c;
    float temperature;
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
        .init_fresh = 0,
        .puct_c = PUCT_C,
        .temperature = 1.0f
    };
}

// =============================================================================
// PRINT HELPERS
// =============================================================================

static int get_omp_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Format large numbers with thousand separators (e.g. 1,234,567)
static const char* format_num(long long n) {
    static char buffers[4][64];
    static int buf_idx = 0;
    char *buf = buffers[buf_idx];
    buf_idx = (buf_idx + 1) % 4;
    
    char temp[64];
    sprintf(temp, "%lld", n);
    int len = (int)strlen(temp);
    int out_idx = 0;
    for (int i = 0; i < len; i++) {
        if (i > 0 && (len - i) % 3 == 0 && temp[i] != '-') {
            buf[out_idx++] = ',';
        }
        buf[out_idx++] = temp[i];
    }
    buf[out_idx] = '\0';
    return buf;
}

// Format seconds into human readable time (e.g. "1m 23s" or "2h 15m")
static const char* format_time(double seconds) {
    static char buf[64];
    if (seconds < 60) {
        sprintf(buf, "%.1fs", seconds);
    } else if (seconds < 3600) {
        int m = (int)(seconds / 60);
        int s = (int)seconds % 60;
        sprintf(buf, "%dm %02ds", m, s);
    } else {
        int h = (int)(seconds / 3600);
        int m = ((int)seconds % 3600) / 60;
        sprintf(buf, "%dh %02dm", h, m);
    }
    return buf;
}

// =============================================================================
// RNG UTILITIES (thread-safe xorshift32)
// =============================================================================


typedef struct { uint32_t state; } RNG;

static inline void rng_seed(RNG *r, uint32_t s) { r->state = s ? s : 1; }

static inline uint32_t rng_u32(RNG *r) {
    uint32_t x = r->state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return r->state = x;
}

static inline float rng_f32(RNG *r) { 
    return (float)rng_u32(r) / (float)UINT32_MAX; 
}

static float rng_gamma(RNG *r, float alpha) {
    if (alpha < 1.0f) return rng_gamma(r, 1.0f + alpha) * powf(rng_f32(r), 1.0f / alpha);
    float d = alpha - 1.0f/3.0f, c = 1.0f / sqrtf(9.0f * d);
    while (1) {
        float x, v;
        do { 
            x = sqrtf(-2.0f * logf(rng_f32(r))) * cosf(2.0f * M_PI * rng_f32(r)); 
            v = 1.0f + c * x; 
        } while (v <= 0);
        v = v * v * v;
        float u = rng_f32(r);
        if (u < 1.0f - 0.0331f * x*x*x*x || logf(u) < 0.5f * x*x + d * (1.0f - v + logf(v))) 
            return d * v;
    }
}

// =============================================================================
// SELFPLAY UTILITIES
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

static int game_over(const GameState *s) { 
    MoveList m; 
    generate_moves(s, &m); 
    return m.count == 0; 
}

// Game ending reasons
typedef enum {
    END_CHECKMATE,      // No legal moves
    END_RESIGNATION,    // Early resignation
    END_MERCY,          // Mercy rule
    END_40_MOVE,        // 40 moves without capture (with ladies)
    END_MAX_MOVES       // Hit 200 move limit
} EndReason;

// Mercy rule configuration
typedef struct {
    int piece_adv_moves;      // After how many moves to check piece advantage
    int piece_adv_threshold;  // How much piece advantage needed
    int lady_dom_moves;       // After how many moves to check lady dominance
    int lady_dom_pawns;       // Max opponent pawns for lady dominance
    int endgame_moves;        // After how many moves to check endgame
    int endgame_pieces;       // Max opponent pieces for endgame win
    int shuffle_moves;        // Moves without capture to trigger shuffle
    int max_total_moves;      // Force adjudication after this many moves
} MercyConfig;

// Preset configurations
static const MercyConfig MERCY_AGGRESSIVE = {
    .piece_adv_moves = 40, .piece_adv_threshold = 2,
    .lady_dom_moves = 30, .lady_dom_pawns = 2,
    .endgame_moves = 60, .endgame_pieces = 3,
    .shuffle_moves = 40, .max_total_moves = 100
};

static const MercyConfig MERCY_INTERMEDIATE = {
    .piece_adv_moves = 50, .piece_adv_threshold = 3,
    .lady_dom_moves = 40, .lady_dom_pawns = 1,
    .endgame_moves = 80, .endgame_pieces = 2,
    .shuffle_moves = 50, .max_total_moves = 150
};

static const MercyConfig MERCY_RELAXED = {
    .piece_adv_moves = 70, .piece_adv_threshold = 4,
    .lady_dom_moves = 60, .lady_dom_pawns = 0,
    .endgame_moves = 100, .endgame_pieces = 1,
    .shuffle_moves = 60, .max_total_moves = 200
};

// Current mercy config (can be changed at runtime)
static MercyConfig current_mercy = {
    .piece_adv_moves = 50, .piece_adv_threshold = 3,
    .lady_dom_moves = 40, .lady_dom_pawns = 1,
    .endgame_moves = 80, .endgame_pieces = 2,
    .shuffle_moves = 50, .max_total_moves = 150
};

static int mercy_rule_check(const GameState *s, int moves, const MercyConfig *cfg) {
    // Count pieces
    int wp = __builtin_popcountll(s->white_pieces);
    int bp = __builtin_popcountll(s->black_pieces);
    int wl = __builtin_popcountll(s->white_ladies);
    int bl = __builtin_popcountll(s->black_ladies);
    int w_total = wp + wl;
    int b_total = bp + bl;
    
    // RULE 1: Piece advantage
    if (moves >= cfg->piece_adv_moves) {
        if (w_total >= b_total + cfg->piece_adv_threshold) return 1;
        if (b_total >= w_total + cfg->piece_adv_threshold) return -1;
    }
    
    // RULE 2: Lady dominance
    if (moves >= cfg->lady_dom_moves) {
        if (wl >= 1 && bl == 0 && bp <= cfg->lady_dom_pawns) return 1;
        if (bl >= 1 && wl == 0 && wp <= cfg->lady_dom_pawns) return -1;
    }
    
    // RULE 3: Endgame advantage
    if (moves >= cfg->endgame_moves) {
        if (wl > bl && b_total <= cfg->endgame_pieces) return 1;
        if (bl > wl && w_total <= cfg->endgame_pieces) return -1;
    }
    
    // RULE 4: Shuffle/long game detection
    if (s->moves_without_captures >= cfg->shuffle_moves || moves >= cfg->max_total_moves) {
        // TIE-BREAKER 1: Material (Lady = 3 pawns)
        int w_mat = wp + (wl * 3);
        int b_mat = bp + (bl * 3);
        
        if (w_mat > b_mat) return 1;
        if (b_mat > w_mat) return -1;
        
        // TIE-BREAKER 2: Positional advancement
        int w_pos = 0, b_pos = 0;
        for (int i = 0; i < 64; i++) {
            int rank = i / 8;
            if ((s->white_pieces >> i) & 1) w_pos += rank;
            if ((s->black_pieces >> i) & 1) b_pos += (7 - rank);
            // Ladies count double for positional value
            if ((s->white_ladies >> i) & 1) w_pos += 4;  // Center value
            if ((s->black_ladies >> i) & 1) b_pos += 4;
        }
        
        if (w_pos > b_pos) return 1;
        if (b_pos > w_pos) return -1;
        
        // TIE-BREAKER 3: Centrality (pieces in center columns are better)
        int w_center = 0, b_center = 0;
        for (int i = 0; i < 64; i++) {
            int col = i % 8;
            int center_bonus = (col >= 2 && col <= 5) ? 1 : 0;  // Columns C-F
            if (((s->white_pieces | s->white_ladies) >> i) & 1) w_center += center_bonus;
            if (((s->black_pieces | s->black_ladies) >> i) & 1) b_center += center_bonus;
        }
        
        if (w_center > b_center) return 1;
        if (b_center > w_center) return -1;
        
        // TIE-BREAKER 4: Pseudo-random based on position hash (50/50 fair)
        // Uses hash bit to avoid systematic bias toward either color
        return (s->hash & 1) ? 1 : -1;
    }
    
    return 0;  // No adjudication
}

// Play one selfplay game, returns result (+1 white, -1 black, 0 draw)
static int play_selfplay_game(int game_id, const CNNWeights *weights, MCTSConfig cfg_cnn, 
                               MCTSConfig cfg_opp, GameStep *history, int *out_len, 
                               EndReason *out_reason, float initial_temp, RNG *rng) {
    GameState state;
    init_game(&state);
    Arena arena;
    arena_init(&arena, 100 * 1024 * 1024);
    
    // Random openings: 2 random moves to create slight imbalance
    for (int i = 0; i < 2; i++) {
        MoveList list;
        generate_moves(&state, &list);
        if (list.count > 0) {
            int random_idx = rng_u32(rng) % list.count;
            apply_move(&state, &list.moves[random_idx]);
        }
    }
    
    int moves = 0;
    float policy[CNN_POLICY_SIZE];
    int hybrid = (game_id % 10 >= 7);  // 70% CNN vs CNN, 30% hybrid
    
    while (!game_over(&state) && moves < DEFAULT_MAX_MOVES) {
        float temp = (moves < DEFAULT_TEMP_THRESHOLD) ? initial_temp : 0.05f;
        
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
        
        if (moves < 30 && is_cnn) {
            mcts_search(root, &arena, TIME_HIGH, cfg, NULL, NULL);
            add_dirichlet_noise(root, rng, DEFAULT_DIRICHLET_EPSILON, DEFAULT_DIRICHLET_ALPHA);
        }
        
        mcts_search(root, &arena, TIME_HIGH, cfg, NULL, NULL);
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
        
        if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
            arena_free(&arena);
            *out_len = moves;
            *out_reason = END_40_MOVE;
            return 0;
        }
        
        int m = mercy_rule_check(&state, moves, &current_mercy);
        if (m == 2) {
            // Early draw (equal material, shuffling)
            arena_free(&arena);
            *out_len = moves;
            *out_reason = END_MERCY;
            return 0;  // Draw
        }
        if (m != 0) {
            arena_free(&arena);
            *out_len = moves;
            *out_reason = END_MERCY;
            return m;  // Win (+1) or Loss (-1)
        }
        
        // Early resignation check
        if (moves > 40) {
            CNNOutput out;
            cnn_forward_with_history(weights, &state, 
                moves >= 1 ? &history[moves-1].state : NULL,
                moves >= 2 ? &history[moves-2].state : NULL, &out);
            if (out.value < -0.85f) {
                arena_free(&arena);
                *out_len = moves;
                *out_reason = END_RESIGNATION;
                return (state.current_player == WHITE) ? -1 : 1;
            }
        }
    }
    
    arena_free(&arena);
    *out_len = moves;
    
    // Determine ending reason
    if (moves >= DEFAULT_MAX_MOVES) {
        *out_reason = END_MAX_MOVES;
        return 0;
    }
    
    // No legal moves = checkmate
    *out_reason = END_CHECKMATE;
    return (state.current_player == WHITE) ? -1 : 1;
}

// =============================================================================
// SELFPLAY MAIN
// =============================================================================

static int run_selfplay(const TrainConfig *cfg) {
    int num_games = cfg->epochs; // Using 'epochs' field to store game count for selfplay
    const char *output_file = cfg->train_file;
    const char *weights_file = cfg->weights_file;
    time_t now = time(NULL);
    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&now));
    
    printf("\n┌────────────────────────────────────────────────────────────────────┐\n");
    printf("│                       SELF-PLAY GENERATION                        │\n");
    printf("├────────────────────────────────────────────────────────────────────┤\n");
    printf("│  Date         : %-50s │\n", time_str);
    printf("│  Games        : %-6d                                             │\n", num_games);
    printf("│  Output       : %-50s │\n", output_file);
    printf("├────────────────────────────────────────────────────────────────────┤\n");
    printf("│  MCTS Nodes   : %-4d (CNN) / %-4d (Opponent)                       │\n", DEFAULT_SELFPLAY_NODES, DEFAULT_SELFPLAY_NODES * 2);
    printf("│  Dirichlet    : α=%.2f, ε=%.2f (first 30 moves)                    │\n", DEFAULT_DIRICHLET_ALPHA, DEFAULT_DIRICHLET_EPSILON);
    printf("│  Temperature  : %.1f → 0.05 (after move %d)                         │\n", cfg->temperature, DEFAULT_TEMP_THRESHOLD);
    printf("│  Openings     : 2 Random Moves                                     │\n");
    printf("│  Opponent     : Grandmaster (same weights)                         │\n");
    printf("├────────────────────────────────────────────────────────────────────┤\n");
    printf("│  Adjudication : Mercy Rule (Intermediate)                          │\n");
    printf("│  Soft Rewards : Checkmate ±1.0 │ Mercy ±0.7 │ Draw 0.0             │\n");
    printf("│  Max Moves    : %d (or 150 with mercy)                             │\n", DEFAULT_MAX_MOVES);
    printf("├────────────────────────────────────────────────────────────────────┤\n");
    printf("│  OMP Threads  : %-2d                                                 │\n", get_omp_threads());
    printf("│  Backend      : Apple Accelerate (BLAS/vDSP)                       │\n");
    printf("└────────────────────────────────────────────────────────────────────┘\n\n");
    
    zobrist_init();
    init_move_tables();
    
    CNNWeights weights;
    cnn_init(&weights);
    if (cnn_load_weights(&weights, weights_file) != 0) {
        printf("⚠ No weights at %s, using random init\n", weights_file);
    } else {
        printf("✓ Loaded weights from %s\n", weights_file);
    }
    
    MCTSConfig cfg_cnn = { 
        .ucb1_c = 1.5, .use_puct = 1, .puct_c = cfg->puct_c, 
        .cnn_weights = (void*)&weights, .max_nodes = DEFAULT_SELFPLAY_NODES 
    };
    
    // Use Grandmaster as opponent for higher quality games
    MCTSConfig cfg_gm = mcts_get_preset(MCTS_PRESET_GRANDMASTER);
    cfg_gm.cnn_weights = (void*)&weights;  // Same weights as CNN player
    cfg_gm.max_nodes = DEFAULT_SELFPLAY_NODES;
    
    MCTSConfig opponents[] = { cfg_gm };  // Single strong opponent
    int num_opps = 1;
    
    remove(output_file);
    
    RNG rng;
    rng_seed(&rng, now);
    
    int batch_cap = 2000;
    TrainingSample *batch = malloc(batch_cap * sizeof(TrainingSample));
    int batch_count = 0, total_samples = 0;
    
    // Track game results
    int game_wins = 0, game_losses = 0, game_draws = 0;
    int *game_lengths = malloc(num_games * sizeof(int));
    int min_len = 9999, max_len = 0;
    int end_counts[5] = {0}; // Track ending reasons
    
    for (int g = 0; g < num_games; g++) {
        GameStep *history = malloc(DEFAULT_MAX_MOVES * sizeof(GameStep));
        int steps = 0;
        EndReason reason;
        
        MCTSConfig opp = opponents[g % num_opps];
        int result = play_selfplay_game(g, &weights, cfg_cnn, opp, history, &steps, &reason, cfg->temperature, &rng);
        total_samples += steps;
        game_lengths[g] = steps;
        if (steps < min_len) min_len = steps;
        if (steps > max_len) max_len = steps;
        end_counts[reason]++;
        
        // Track results
        if (result > 0) game_wins++;
        else if (result < 0) game_losses++;
        else game_draws++;
        
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
            // Soft reward: checkmate = 1.0, mercy = 0.7, draw = 0
            float value_scale = (reason == END_CHECKMATE || reason == END_RESIGNATION) ? 1.0f : 0.7f;
            s->target_value = (result == 0) ? 0.0f : value_scale * (float)(result * sign);
        }
        
        free(history);
        
        if (g % 10 == 0) {
            printf("\rGames: %d/%d | W:%d L:%d D:%d | Samples: %d", 
                   g, num_games, game_wins, game_losses, game_draws, total_samples);
            fflush(stdout);
        }
    }
    
    if (batch_count > 0) dataset_save_append(output_file, batch, batch_count);
    free(batch);
    cnn_free(&weights);
    
    // Compute stats
    float mean = (float)total_samples / num_games;
    float variance = 0;
    for (int g = 0; g < num_games; g++) {
        float diff = game_lengths[g] - mean;
        variance += diff * diff;
    }
    variance /= num_games;
    float std_dev = sqrtf(variance);
    free(game_lengths);
    
    printf("\n\n=== Self-Play Results ===\n");
    printf("  Games: %d\n", num_games);
    printf("  Length: %.1f ± %.1f (min: %d, max: %d)\n", mean, std_dev, min_len, max_len);
    printf("  White Wins: %d (%.1f%%)\n", game_wins, 100.0f * game_wins / num_games);
    printf("  Black Wins: %d (%.1f%%)\n", game_losses, 100.0f * game_losses / num_games);
    printf("  Draws:      %d (%.1f%%)\n", game_draws, 100.0f * game_draws / num_games);
    printf("\n  End Reasons:\n");
    printf("    Checkmate:   %d\n", end_counts[END_CHECKMATE]);
    printf("    Resignation: %d\n", end_counts[END_RESIGNATION]);
    printf("    Mercy Rule:  %d\n", end_counts[END_MERCY]);
    printf("    40-Move:     %d\n", end_counts[END_40_MOVE]);
    printf("    Max Moves:   %d\n", end_counts[END_MAX_MOVES]);
    printf("\nGenerated %d samples.\n", total_samples);
    return 0;
}

// =============================================================================
// TRAINING UTILITIES
// =============================================================================

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

static void print_dataset_stats(TrainingSample *data, size_t count, BalancedIndex *idx) {
    printf("\n=== Dataset Statistics ===\n");
    printf("  Natural Dist: Wins %'d (%.1f%%) | Losses %'d (%.1f%%) | Draws %'d (%.1f%%)\n", 
           idx->w_cnt, (float)idx->w_cnt/count*100, 
           idx->l_cnt, (float)idx->l_cnt/count*100, 
           idx->d_cnt, (float)idx->d_cnt/count*100);
    
    // Count samples with ladies
    int with_ladies = 0;
    for (int i = 0; i < (int)count; i++) {
        if (data[i].state.white_ladies || data[i].state.black_ladies) {
            with_ladies++;
        }
    }
    printf("  Samples with Ladies: %'d (%.1f%%)\n", with_ladies, (float)with_ladies/count*100);
    
    // Sampling strategy
    float nat_draw_pct = (float)idx->d_cnt / count * 100;
    float target_draw_pct = (nat_draw_pct < 10.0f) ? nat_draw_pct : 10.0f;
    float target_wl_pct = (100.0f - target_draw_pct) / 2.0f;
    printf("  Target Dist : Wins %.1f%% | Losses %.1f%% | Draws %.1f%%\n", 
           target_wl_pct, target_wl_pct, target_draw_pct);
}

static void print_training_header(int epochs, float lr, int batch_size, float l2_decay, int patience) {
    // Calculate network size
    int conv_params = 64 * CNN_INPUT_CHANNELS * 9 + 64;  // conv1
    conv_params += 3 * (64 * 64 * 9 + 64);                // conv2-4
    int bn_params = 4 * 64 * 4;                           // 4 BN layers
    int policy_params = 512 * 4097 + 512;                  // policy head
    int value_params = 256 * 4097 + 256 + 256 + 1;        // value head
    int total_params = conv_params + bn_params + policy_params + value_params;
    
    printf("\n┌────────────────────────────────────────────────────────────────────┐\n");
    printf("│                        CNN TRAINING CONFIG                         │\n");
    printf("├────────────────────────────────────────────────────────────────────┤\n");
    printf("│  Network      : 4 Conv (64ch) + Policy (512) + Value (256→1)       │\n");
    printf("│  Parameters   : %s (~%.1f MB)                               │\n", format_num(total_params), total_params * 4.0f / 1024 / 1024);
    printf("├────────────────────────────────────────────────────────────────────┤\n");
    printf("│  Epochs       : %-4d            Batch Size : %-4d                  │\n", epochs, batch_size);
    printf("│  Learning Rate: %.6f        L2 Decay   : %.1e                │\n", lr, l2_decay);
    printf("│  LR Warmup    : Epoch 1 ramp   Early Stop : %d epochs patience     │\n", patience);
    printf("├────────────────────────────────────────────────────────────────────┤\n");
    printf("│  Rewards      : Checkmate ±1.0 │ Mercy ±0.7 │ Draw 0.0             │\n");
    printf("│  Canonical    : Board flipped for Black (always \"my turn\")         │\n");
    printf("├────────────────────────────────────────────────────────────────────┤\n");
    printf("│  OMP Threads  : %-2d               Backend: Apple Accelerate         │\n", get_omp_threads());
    printf("└────────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("+-------+---------------------------+---------------------------+------------+\n");
    printf("| Epoch |        Train Loss         |         Val Loss          |   Status   |\n");
    printf("|       |  Total  | Policy  | Value |  Total  | Policy  | Value |            |\n");
    printf("+-------+---------+---------+-------+---------+---------+-------+------------+\n");
}

static void print_epoch_result(int epoch, float et, float ep, float ev, 
                               LossMetrics val, int is_best, int patience, int max_patience) {
    if (is_best) {
        printf("|  %3d  | %7.4f | %7.4f | %5.3f | %7.4f | %7.4f | %5.3f |   *BEST*   |\n",
               epoch, et, ep, ev, val.total, val.policy, val.value);
    } else {
        printf("|  %3d  | %7.4f | %7.4f | %5.3f | %7.4f | %7.4f | %5.3f |  wait %d/%d |\n",
               epoch, et, ep, ev, val.total, val.policy, val.value, patience, max_patience);
    }
}

// =============================================================================
// TRAINING MAIN
// =============================================================================

static int run_training(TrainConfig *cfg) {
    // Header with timestamp
    time_t now = time(NULL);
    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&now));
    printf("=== CNN Training [%s] ===\n\n", time_str);
    
    srand((unsigned)now);
    zobrist_init();
    init_move_tables();
    
    // Load data
    printf("Loading: %s\n", cfg->train_file);
    int total = 0;
    TrainingSample *all_data = load_dataset_file(cfg->train_file, &total);
    if (!all_data) return 1;
    
    DatasetSplit split = split_dataset(all_data, total, cfg->val_file, 0.90f);
    BalancedIndex train_idx = build_balanced_index(split.train_data, split.train_count);
    BalancedIndex val_idx = build_balanced_index(split.val_data, split.val_count);
    
    // Print dataset stats
    print_dataset_stats(split.train_data, split.train_count, &train_idx);
    if (val_idx.w_cnt + val_idx.l_cnt + val_idx.d_cnt > 0) {
        printf("  Val Dist    : Wins %'d | Losses %'d | Draws %'d\n", 
               val_idx.w_cnt, val_idx.l_cnt, val_idx.d_cnt);
    }
    
    TrainingSample *batch = malloc(cfg->batch_size * sizeof(TrainingSample));
    
    // Verify balanced sampling with one epoch worth of batches
    int nb = split.train_count / cfg->batch_size;
    int batch_wins = 0, batch_losses = 0, batch_draws = 0;
    for (int b = 0; b < nb; b++) {
        fill_balanced_batch(batch, cfg->batch_size, split.train_data, &train_idx, split.train_count);
        for (int k = 0; k < cfg->batch_size; k++) {
            if (batch[k].target_value > 0.1f) batch_wins++;
            else if (batch[k].target_value < -0.1f) batch_losses++;
            else batch_draws++;
        }
    }
    int total_batch = batch_wins + batch_losses + batch_draws;
    printf("  Sampling   : W %'d (%.1f%%) | L %'d (%.1f%%) | D %'d (%.1f%%)\n",
           batch_wins, 100.0f * batch_wins / total_batch,
           batch_losses, 100.0f * batch_losses / total_batch,
           batch_draws, 100.0f * batch_draws / total_batch);
    
    // Init weights
    CNNWeights weights;
    cnn_init(&weights);
    if (!cfg->init_fresh && cnn_load_weights(&weights, cfg->weights_file) == 0)
        printf("\n✓ Loaded weights from %s\n", cfg->weights_file);
    else
        printf("\n⚠ Starting fresh\n");
    
    // Training header
    print_training_header(cfg->epochs, cfg->learning_rate, cfg->batch_size, cfg->l2_decay, cfg->patience);
    
    float lr = cfg->learning_rate, best = 1e9f;
    int patience = 0;
    time_t training_start = time(NULL);
    int total_samples_processed = 0;
    
    // Training loop
    for (int epoch = 1; epoch <= cfg->epochs; epoch++) {
        float et = 0, ep = 0, ev = 0;
        int num_batches = split.train_count / cfg->batch_size;
        
        for (int b = 0; b < num_batches; b++) {
            fill_balanced_batch(batch, cfg->batch_size, split.train_data, &train_idx, split.train_count);
            float eff_lr = (epoch == 1) ? lr * (b+1.0f) / num_batches : lr;
            float p, v;
            et += cnn_train_step(&weights, batch, cfg->batch_size, eff_lr, 0, cfg->l2_decay, &p, &v);
            ep += p;
            ev += v;
            total_samples_processed += cfg->batch_size;
            
            // Print progress every 100 batches
            if ((b + 1) % 100 == 0 || b == num_batches - 1) {
                float avg_t = et / (b + 1), avg_p = ep / (b + 1), avg_v = ev / (b + 1);
                printf("\r  Batch %d/%d | Loss: %.4f (P:%.4f V:%.4f)    ", 
                       b + 1, num_batches, avg_t, avg_p, avg_v);
                fflush(stdout);
            }
        }
        printf("\n");  // Newline after batch loop
        et /= num_batches; ep /= num_batches; ev /= num_batches;
        
        LossMetrics val = run_validation(&weights, split.val_data, &val_idx, split.val_count, batch, cfg->batch_size);
        
        int is_best = (val.total < best);
        if (is_best) {
            best = val.total;
            cnn_save_weights(&weights, cfg->weights_file);
            patience = 0;
        } else {
            patience++;
        }
        
        print_epoch_result(epoch, et, ep, ev, val, is_best, patience, cfg->patience);
        
        // LR decay on plateau
        if (patience >= cfg->patience) {
            lr *= 0.1f;
            if (lr < 1e-6f) { printf("\nConverged!\n"); break; }
            printf("*** LR -> %.6f ***\n", lr);
            patience = 0;
        }
    }
    
    double training_time = difftime(time(NULL), training_start);
    double samples_per_sec = (training_time > 0) ? total_samples_processed / training_time : 0;
    
    printf("+-------+---------+---------+-------+---------+---------+-------+------------+\n");
    printf("\n┌────────────────────────────────────────────────────────────────────┐\n");
    printf("│                       TRAINING COMPLETE                            │\n");
    printf("├────────────────────────────────────────────────────────────────────┤\n");
    printf("│  Best Loss    : %.4f                                              │\n", best);
    printf("│  Weights      : %-50s │\n", cfg->weights_file);
    printf("│  Time         : %-20s                               │\n", format_time(training_time));
    printf("│  Throughput   : %s samples/sec                             │\n", format_num((long long)samples_per_sec));
    printf("└────────────────────────────────────────────────────────────────────┘\n");
    
    // Cleanup
    free(all_data); 
    free(batch);
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

static void print_help(void) {
    printf("Usage: dama train [options]\n\n");
    printf("Self-play:\n");
    printf("  --selfplay        Generate data before training\n");
    printf("  --games N         Self-play games (default: 1000)\n");
    printf("\nTraining:\n");
    printf("  --epochs N        Training epochs (default: %d)\n", DEFAULT_EPOCHS);
    printf("  --lr RATE         Learning rate (default: %.4f)\n", DEFAULT_LR);
    printf("  --batch N         Batch size (default: %d)\n", DEFAULT_BATCH_SIZE);
    printf("  --puct F          MCTS PUCT constant (default: %.2f)\n", PUCT_C);
    printf("  --temp F          Sampling temperature (default: 1.0)\n");
    printf("  --data FILE       Training data file\n");
    printf("  --weights FILE    Weights file\n");
    printf("  --init            Fresh weights\n");
}

int cmd_train(int argc, char **argv) {
    setlocale(LC_NUMERIC, "");
    
    TrainConfig cfg = default_config();
    int do_selfplay = 0, selfplay_games = 1000;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_help();
            return 0;
        }
        else if (strcmp(argv[i], "--selfplay") == 0) do_selfplay = 1;
        else if (strcmp(argv[i], "--games") == 0 && i+1 < argc) selfplay_games = atoi(argv[++i]);
        else if (strcmp(argv[i], "--epochs") == 0 && i+1 < argc) cfg.epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) cfg.learning_rate = atof(argv[++i]);
        else if (strcmp(argv[i], "--puct") == 0 && i+1 < argc) cfg.puct_c = atof(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i+1 < argc) cfg.temperature = atof(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i+1 < argc) cfg.batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--data") == 0 && i+1 < argc) cfg.train_file = argv[++i];
        else if (strcmp(argv[i], "--weights") == 0 && i+1 < argc) cfg.weights_file = argv[++i];
        else if (strcmp(argv[i], "--init") == 0) cfg.init_fresh = 1;
    }
    
    // Create log file with timestamp
    mkdir(DEFAULT_LOG_DIR, 0755);
    time_t now = time(NULL);
    char log_filename[256];
    strftime(log_filename, sizeof(log_filename), DEFAULT_LOG_DIR "/train_%Y%m%d_%H%M%S.log", localtime(&now));
    g_logfile = fopen(log_filename, "w");
    if (g_logfile) {
        log_printf("=== Training Log: %s ===\n\n", log_filename);
    }
    
    // Store actual epochs for training before selfplay overwrites
    int actual_epochs = cfg.epochs;
    
    if (do_selfplay) {
        // Temporarily use epochs field for number of games in selfplay
        cfg.epochs = selfplay_games;
        if (run_selfplay(&cfg) != 0)
            return 1;
        // Restore actual epochs for training
        cfg.epochs = actual_epochs;
    }
    
    // Skip training if epochs is 0
    if (cfg.epochs <= 0) {
        log_printf("Skipping training (epochs = %d)\n", cfg.epochs);
        if (g_logfile) { fclose(g_logfile); g_logfile = NULL; }
        return 0;
    }
    
    int result = run_training(&cfg);
    
    // Close log file
    if (g_logfile) {
        log_printf("\n=== Log saved to: %s ===\n", log_filename);
        fclose(g_logfile);
        g_logfile = NULL;
    }
    
    return result;
}
/**
 * cmd_clop.c - CLOP Tuning CLI Command
 * 
 * Usage: dama clop [options]
 * 
 * Options:
 *   --iterations N  Number of CLOP iterations (default: 50)
 *   --games N       Games per evaluation (default: 10)
 *   --nodes N       MCTS nodes per move (default: 200)
 *   --seed N        Random seed (default: time-based)
 *   --verbose       Print detailed output
 *   --help          Show this help
 */

#include "dama/tuning/clop.h"
#include "dama/tuning/clop_params.h"
#include "dama/tournament/tournament.h"
#include "dama/search/mcts.h"
#include "dama/search/mcts_config.h"
#include "dama/engine/game.h"
#include "dama/engine/movegen.h"
#include "dama/common/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// =============================================================================
// HELPER: Apply CLOP params to MCTSConfig
// =============================================================================

static void apply_clop_params(MCTSConfig *cfg, const double *params) {
    // Map CLOP params to MCTSConfig fields (must match CLOP_DEFAULT_PARAMS order)
    cfg->ucb1_c = params[0];
    cfg->puct_c = params[1];
    cfg->bias_constant = params[2];
    cfg->fpu_value = params[3];
    cfg->decay_factor = params[4];
    cfg->weights.w_capture = params[5];
    cfg->weights.w_promotion = params[6];
    cfg->weights.w_advance = params[7];
    cfg->weights.w_center = params[8];
    cfg->weights.w_edge = params[9];
    cfg->weights.w_threat = params[10];
    cfg->weights.w_lady_activity = params[11];
}

// =============================================================================
// HELPER: Evaluate Parameters via Mini-Tournament
// =============================================================================

static double evaluate_params(const double *params, int games, int nodes) {
    // Create two players: tuned vs baseline
    TournamentPlayer players[2];
    memset(players, 0, sizeof(players));
    
    // Player 0: Tuned
    strcpy(players[0].name, "Tuned");
    players[0].config = mcts_get_preset(MCTS_PRESET_GRANDMASTER);
    players[0].config.max_nodes = nodes;
    apply_clop_params(&players[0].config, params);
    
    // Player 1: Baseline (default Grandmaster)
    strcpy(players[1].name, "Baseline");
    players[1].config = mcts_get_preset(MCTS_PRESET_GRANDMASTER);
    players[1].config.max_nodes = nodes;
    
    // Run mini-tournament
    TournamentSystemConfig tcfg;
    memset(&tcfg, 0, sizeof(tcfg));
    tcfg.num_players = 2;
    tcfg.players = players;
    tcfg.games_per_pair = games;
    tcfg.time_limit = 0;  // Use max_nodes instead
    tcfg.parallel_games = 0;
    
    tournament_run(&tcfg);
    
    // Return win rate of tuned player
    int total = players[0].wins + players[0].losses + players[0].draws;
    if (total == 0) return 0.5;
    
    double win_rate = (players[0].wins + 0.5 * players[0].draws) / total;
    return win_rate;
}

// =============================================================================
// MAIN COMMAND
// =============================================================================

int cmd_clop(int argc, char **argv) {
    int iterations = 50;
    int games = 10;
    int nodes = 200;
    int verbose = 0;
    unsigned int seed = (unsigned int)time(NULL);
    
    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--games") == 0 && i + 1 < argc) {
            games = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nodes") == 0 && i + 1 < argc) {
            nodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: dama clop [options]\n\n");
            printf("Options:\n");
            printf("  --iterations N  Number of CLOP iterations (default: 50)\n");
            printf("  --games N       Games per evaluation (default: 10)\n");
            printf("  --nodes N       MCTS nodes per move (default: 200)\n");
            printf("  --seed N        Random seed\n");
            printf("  --verbose       Print detailed output\n");
            printf("  --help          Show this help\n");
            return 0;
        }
    }
    
    // Initialize game tables
    zobrist_init();
    movegen_init();
    
    log_printf("\n");
    log_printf("┌────────────────────────────────────────────────────────────────────┐\n");
    log_printf("│                        CLOP PARAMETER TUNING                       │\n");
    log_printf("├────────────────────────────────────────────────────────────────────┤\n");
    log_printf("│  Iterations : %-6d      Games/Eval : %-6d                       │\n", iterations, games);
    log_printf("│  MCTS Nodes : %-6d      Random Seed: %-10u                   │\n", nodes, seed);
    log_printf("│  Parameters : %-6d                                               │\n", (int)CLOP_DEFAULT_NUM_PARAMS);
    log_printf("└────────────────────────────────────────────────────────────────────┘\n\n");
    
    // Initialize CLOP
    CLOPState clop;
    if (clop_init(&clop, CLOP_DEFAULT_PARAMS, CLOP_DEFAULT_NUM_PARAMS, iterations) < 0) {
        log_error("Failed to initialize CLOP");
        return 1;
    }
    clop.rng_state = seed;
    
    double *params = malloc(CLOP_DEFAULT_NUM_PARAMS * sizeof(double));
    if (!params) {
        clop_free(&clop);
        return 1;
    }
    
    // Main CLOP loop
    for (int iter = 0; iter < iterations; iter++) {
        log_printf("[CLOP] Iteration %d/%d\n", iter + 1, iterations);
        
        // Get next params to try
        if (clop_suggest(&clop, params) < 0) {
            log_error("clop_suggest failed");
            break;
        }
        
        if (verbose) {
            log_printf("  Params: ");
            for (int i = 0; i < (int)CLOP_DEFAULT_NUM_PARAMS; i++) {
                log_printf("%.3f ", params[i]);
            }
            log_printf("\n");
        }
        
        // Evaluate
        double win_rate = evaluate_params(params, games, nodes);
        log_printf("  Win rate: %.1f%%\n", win_rate * 100.0);
        
        // Update CLOP
        if (clop_update(&clop, params, win_rate) < 0) {
            log_error("clop_update failed");
            break;
        }
    }
    
    // Print best parameters
    clop_get_best(&clop, params);
    
    log_printf("\n");
    log_printf("┌────────────────────────────────────────────────────────────────────┐\n");
    log_printf("│                        BEST PARAMETERS FOUND                       │\n");
    log_printf("├────────────────────────────────────────────────────────────────────┤\n");
    for (int i = 0; i < (int)CLOP_DEFAULT_NUM_PARAMS; i++) {
        log_printf("│  %-12s : %-10.4f  [%.2f - %.2f]                          │\n",
               CLOP_DEFAULT_PARAMS[i].name, params[i],
               CLOP_DEFAULT_PARAMS[i].lower, CLOP_DEFAULT_PARAMS[i].upper);
    }
    log_printf("├────────────────────────────────────────────────────────────────────┤\n");
    log_printf("│  Best Win Rate: %.1f%%                                             │\n", 
               clop.best_outcome * 100.0);
    log_printf("└────────────────────────────────────────────────────────────────────┘\n");
    
    free(params);
    clop_free(&clop);
    
    return 0;
}

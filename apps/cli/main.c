/**
 * dama - Unified CLI Tool for MCTS Dama
 * 
 * Usage:
 *   dama train [options]      - Train CNN (selfplay + training)
 *   dama tournament [options] - Run MCTS tournament
 *   dama data <subcommand>    - Data utilities (inspect, merge)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =============================================================================
// COMMAND IMPLEMENTATIONS (include .c files directly for single-binary build)
// NOTE: cli_view.c first (shared views), then cmd_data.c (provides DatasetSplit)
// =============================================================================

// cli_view.c now compiled as part of LIB_SRCS, just include the header
#include "dama/common/cli_view.h"

// External command handlers (compiled separately via CLI_SRCS in Makefile)
extern int cmd_data(int argc, char **argv);
extern int cmd_train(int argc, char **argv);
extern int cmd_tournament(int argc, char **argv);
extern int cmd_diagnose(int argc, char **argv);
extern int cmd_clop(int argc, char **argv);


// =============================================================================
// COMMAND REGISTRY
// =============================================================================

typedef struct {
    const char *name;
    const char *description;
    int (*handler)(int argc, char **argv);
} Command;

static Command commands[] = {
    {"train",      "Train CNN (selfplay + SGD)",         cmd_train},
    {"tournament", "Run MCTS tournament",                cmd_tournament},
    {"data",       "Data utilities (inspect, merge)",    cmd_data},
    {"diagnose",   "CNN training diagnostics",           cmd_diagnose},
    {"clop",       "CLOP hyperparameter tuning",         cmd_clop},
    {NULL, NULL, NULL}
};

// =============================================================================
// HELP
// =============================================================================

static void print_usage(const char *program) {
    printf("MCTS Dama - Unified CLI Tool\n\n");
    printf("Usage: %s <command> [options]\n\n", program);
    printf("Commands:\n");
    
    for (int i = 0; commands[i].name != NULL; i++) {
        printf("  %-12s  %s\n", commands[i].name, commands[i].description);
    }
    
    printf("\nRun '%s <command> --help' for command-specific options.\n", program);
    printf("\nFor GUI play: ./bin/game_gui\n");
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 0;
    }
    
    const char *cmd_name = argv[1];
    
    if (strcmp(cmd_name, "--help") == 0 || strcmp(cmd_name, "-h") == 0) {
        print_usage(argv[0]);
        return 0;
    }
    
    for (int i = 0; commands[i].name != NULL; i++) {
        if (strcmp(cmd_name, commands[i].name) == 0) {
            return commands[i].handler(argc - 1, argv + 1);
        }
    }

    printf("Error: Unknown command '%s'\n\n", cmd_name);
    print_usage(argv[0]);
    return 1;
}

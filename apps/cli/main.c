/**
 * dama - Unified CLI Tool for MCTS Dama
 * 
 * Usage:
 *   dama train [options]      - Train CNN (selfplay + training)
 *   dama tournament [options] - Run MCTS tournament
 *   dama tune [options]       - SPSA tuning
 *   dama data <subcommand>    - Data utilities (inspect, merge)
 *   dama play [options]       - Play vs AI
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =============================================================================
// COMMAND IMPLEMENTATIONS (include .c files directly for single-binary build)
// =============================================================================

#include "cmd_train.c"

// Stub declarations for not-yet-implemented commands
int cmd_tournament(int argc, char **argv);
int cmd_tune(int argc, char **argv);
int cmd_data(int argc, char **argv);
int cmd_play(int argc, char **argv);

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
    {"tune",       "SPSA hyperparameter tuning",         cmd_tune},
    {"data",       "Data utilities (inspect, merge)",    cmd_data},
    {"play",       "Play vs AI in terminal",             cmd_play},
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

// =============================================================================
// STUB IMPLEMENTATIONS (to be replaced with real implementations)
// =============================================================================

int cmd_tournament(int argc, char **argv) {
    (void)argc; (void)argv;
    printf("TODO: Tournament not yet integrated into unified CLI.\n");
    printf("Use: ./bin/tournament directly for now.\n");
    return 0;
}

int cmd_tune(int argc, char **argv) {
    (void)argc; (void)argv;
    printf("TODO: SPSA tuner not yet integrated into unified CLI.\n");
    return 0;
}

int cmd_data(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: dama data <subcommand>\n");
        printf("  inspect <file>    Show dataset statistics\n");
        printf("  merge <files...>  Merge multiple datasets\n");
        return 1;
    }
    printf("TODO: Data utilities not yet integrated.\n");
    return 0;
}

int cmd_play(int argc, char **argv) {
    (void)argc; (void)argv;
    printf("TODO: Terminal play not yet implemented.\n");
    printf("Use: ./bin/game_gui for GUI play.\n");
    return 0;
}

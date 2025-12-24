UNAME_S := $(shell uname -s)

# Use Clang on macOS for Accelerate framework compatibility
ifeq ($(UNAME_S),Darwin)
    CC = clang
    LIBOMP_PREFIX = /opt/homebrew/opt/libomp
    CFLAGS = -Wall -Wextra -std=c99 -O3 -ffast-math -mcpu=apple-m2 -flto -funroll-loops -MMD -MP -Isrc -Isrc/core -Isrc/mcts -Isrc/nn 
    CFLAGS += -Xclang -fopenmp -I$(LIBOMP_PREFIX)/include -DACCELERATE_NEW_LAPACK
    LDFLAGS = -lm -L$(LIBOMP_PREFIX)/lib -lomp -framework Accelerate
else
    CC = gcc
    CFLAGS = -Wall -Wextra -std=c99 -O3 -march=native -flto -MMD -MP -Isrc -Isrc/core -Isrc/mcts -Isrc/nn 
    CFLAGS += -fopenmp
    LDFLAGS = -lm -fopenmp
endif

OBJ_DIR = obj
BIN_DIR = bin

# =============================================================================
# SOURCE FILES
# =============================================================================

# Core module
CORE_SRCS = src/core/game.c src/core/movegen.c

# MCTS module
MCTS_SRCS = src/mcts/mcts_internal.c  src/mcts/selection.c \
            src/mcts/expansion.c src/mcts/simulation.c src/mcts/backprop.c \
            src/mcts/mcts_presets.c src/mcts/mcts.c

# NN module
NN_SRCS = src/nn/dataset.c src/nn/cnn_core.c src/nn/cnn_inference.c src/nn/cnn_training.c src/nn/conv_ops.c

# All library sources
LIB_SRCS = $(CORE_SRCS) $(MCTS_SRCS) $(NN_SRCS)

# Object files
LIB_OBJS = $(LIB_SRCS:%.c=$(OBJ_DIR)/%.o)
DEPS = $(LIB_OBJS:.o=.d)

# =============================================================================
# TARGETS
# =============================================================================

TARGET = $(BIN_DIR)/dama

all: $(BIN_DIR) $(OBJ_DIR) $(TARGET)

# Convenience alias
main: all
.PHONY: main

# Main CLI game
$(TARGET): $(OBJ_DIR)/apps/cli/dama_cli.o $(LIB_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/apps/cli/dama_cli.o: apps/cli/dama_cli.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Tournament (Combined CNN/Legacy)
tournament: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/tournament tools/evaluation/tournament.c $(LIB_SRCS) $(LDFLAGS)

# SPSA Tuner
tuner: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/tuner tools/evaluation/tuner.c $(LIB_SRCS) $(LDFLAGS)

# Legacy NN Trainer
trainer_legacy: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/trainer_legacy tools/training/trainer_legacy.c $(LIB_SRCS) $(LDFLAGS)

# CNN Trainer
trainer: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/trainer tools/training/trainer.c $(LIB_SRCS) $(LDFLAGS)

# CNN Debugger
cnn_debug: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/cnn_debug test/util/cnn_debug.c $(LIB_SRCS) $(LDFLAGS)

# CNN Overfit Test
cnn_overfit: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/cnn_overfit test/integration/cnn_overfit.c $(LIB_SRCS) $(LDFLAGS)

# Dataset Validator
validator: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/validator test/util/dataset_validator.c $(LIB_SRCS) $(LDFLAGS)

# Self-Play Generator
selfplay: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/selfplay tools/training/selfplay.c $(LIB_SRCS) $(LDFLAGS)

# Dataset Merger
merger: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/merger tools/data/merger.c src/nn/dataset.c $(LDFLAGS)

# Data Inspector
inspector: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/inspector tools/data/inspector.c $(LIB_SRCS) $(LDFLAGS)

# Debug Flip Test
debug_flip: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/debug_flip test/util/debug_flip.c $(LIB_SRCS) $(LDFLAGS)

# Data Generator
bootstrap: $(BIN_DIR) $(OBJ_DIR)
	@mkdir -p data
	$(CC) $(CFLAGS) -o $(BIN_DIR)/bootstrap tools/training/bootstrap.c $(LIB_SRCS) $(LDFLAGS)

# Init Weights (Helper)
init_weights: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/init_weights tools/training/init_weights.c $(LIB_SRCS) $(LDFLAGS)

# 1v1 Comparison
compare: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/compare tools/evaluation/compare.c $(LIB_SRCS) $(LDFLAGS)

# Alias for backward compatibility
fast: compare

# SDL2 GUI
SDL_CFLAGS := $(shell pkg-config --cflags sdl2 2>/dev/null || echo "-I/opt/homebrew/include -I/usr/local/include")
SDL_LDFLAGS := $(shell pkg-config --libs sdl2 2>/dev/null || echo "-L/opt/homebrew/lib -L/usr/local/lib -lSDL2")

gui: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -o $(BIN_DIR)/game_gui apps/gui/dama_gui.c $(LIB_SRCS) $(LDFLAGS) $(SDL_LDFLAGS)

# Tests
tests: $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/run_tests test/unit/test_game.c $(LIB_SRCS) $(LDFLAGS)

# =============================================================================
# BUILD RULES
# =============================================================================

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Pattern rule for object files
$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

-include $(DEPS)

.PHONY: all clean tournament tuner trainer selfplay merger inspector bootstrap fast gui tests
overfit: $(BIN_DIR)/trainer_overfit
.PHONY: overfit

$(BIN_DIR)/trainer_overfit: tools/training/trainer_overfit.c $(LIB_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)


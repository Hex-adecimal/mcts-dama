UNAME_S := $(shell uname -s)

# Use Clang on macOS for Accelerate framework compatibility
ifeq ($(UNAME_S),Darwin)
    CC = clang
    LIBOMP_PREFIX = /opt/homebrew/opt/libomp
    CFLAGS = -Wall -Wextra -std=c99 -O3 -ffast-math -mcpu=apple-m2 -flto -funroll-loops -MMD -MP -Isrc -Isrc/core -Isrc/mcts -Isrc/nn -Isrc/ui
    CFLAGS += -Xclang -fopenmp -I$(LIBOMP_PREFIX)/include -DACCELERATE_NEW_LAPACK
    LDFLAGS = -lm -L$(LIBOMP_PREFIX)/lib -lomp -framework Accelerate
else
    CC = gcc
    CFLAGS = -Wall -Wextra -std=c99 -O3 -march=native -flto -MMD -MP -Isrc -Isrc/core -Isrc/mcts -Isrc/nn -Isrc/ui 
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

# MCTS module (consolidated from 7 files to 3)
MCTS_SRCS = src/mcts/mcts.c src/mcts/mcts_tree.c src/mcts/mcts_rollout.c src/mcts/tournament.c

# NN module
NN_SRCS = src/nn/dataset.c src/nn/dataset_analysis.c src/nn/selfplay.c src/nn/training_pipeline.c src/nn/cnn_core.c src/nn/cnn_inference.c src/nn/cnn_training.c src/nn/conv_ops.c

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

# Unified CLI (main binary)
CLI_SRCS = src/ui/cli_view.c apps/cli/cmd_data.c apps/cli/cmd_train.c apps/cli/cmd_tournament.c
$(TARGET): $(BIN_DIR) $(OBJ_DIR) apps/cli/main.c $(CLI_SRCS)
	$(CC) $(CFLAGS) -o $@ apps/cli/main.c $(LIB_SRCS) $(LDFLAGS)

# Tournament (legacy tool)
tournament: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/tournament legacy/tools/evaluation/tournament.c $(LIB_SRCS) $(LDFLAGS)

# SDL2 GUI
SDL_CFLAGS := $(shell pkg-config --cflags sdl2 2>/dev/null || echo "-I/opt/homebrew/include -I/usr/local/include")
SDL_LDFLAGS := $(shell pkg-config --libs sdl2 2>/dev/null || echo "-L/opt/homebrew/lib -L/usr/local/lib -lSDL2")

gui: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -o $(BIN_DIR)/game_gui _apps/gui/dama_gui.c $(LIB_SRCS) $(LDFLAGS) $(SDL_LDFLAGS)

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

.PHONY: all clean tournament gui tests

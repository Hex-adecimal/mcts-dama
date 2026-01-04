UNAME_S := $(shell uname -s)

# Use Clang on macOS for Accelerate framework compatibility
ifeq ($(UNAME_S),Darwin)
    CC = clang
    LIBOMP_PREFIX = /opt/homebrew/opt/libomp
    CFLAGS = -Wall -Wextra -std=c99 -O3 -ffast-math -mcpu=apple-m2 -flto -funroll-loops -MMD -MP -Iinclude
    CFLAGS += -Xclang -fopenmp -I$(LIBOMP_PREFIX)/include -DACCELERATE_NEW_LAPACK
    LDFLAGS = -lm -L$(LIBOMP_PREFIX)/lib -lomp -framework Accelerate
else
    CC = gcc
    CFLAGS = -Wall -Wextra -std=c99 -O3 -march=native -flto -MMD -MP -Iinclude
    CFLAGS += -fopenmp
    LDFLAGS = -lm -fopenmp
endif

OBJ_DIR = obj
BIN_DIR = bin

# =============================================================================
# SOURCE FILES (New Structure)
# =============================================================================

# Engine module (ex core/)
ENGINE_SRCS = src/engine/game.c src/engine/movegen.c src/engine/endgame.c

# Search module (ex mcts/)
SEARCH_SRCS = src/search/mcts_search.c src/search/mcts_utils.c src/search/mcts_tree.c src/search/mcts_selection.c src/search/mcts_rollout.c src/search/mcts_worker.c

# Neural module (inference only, ex part of nn/)
NEURAL_SRCS = src/neural/cnn_core.c src/neural/cnn_io.c src/neural/cnn_inference.c src/neural/conv_ops.c src/neural/cnn_batch_norm.c src/neural/cnn_encode.c

# Training module (training pipeline, ex part of nn/)
TRAINING_SRCS = src/training/cnn_training.c src/training/dataset.c src/training/dataset_analysis.c src/training/selfplay.c src/training/training_pipeline.c

# Tournament module (moved from src/mcts/ to apps/tournament/)
TOURNAMENT_SRCS = apps/tournament/tournament.c

# All library sources
LIB_SRCS = $(ENGINE_SRCS) $(SEARCH_SRCS) $(NEURAL_SRCS) $(TRAINING_SRCS) $(TOURNAMENT_SRCS)

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
$(TARGET): $(BIN_DIR) $(OBJ_DIR) apps/cli/main.c
	$(CC) $(CFLAGS) -o $@ apps/cli/main.c $(LIB_SRCS) $(LDFLAGS)

# SDL2 GUI
SDL_CFLAGS := $(shell pkg-config --cflags sdl2 2>/dev/null || echo "-I/opt/homebrew/include -I/usr/local/include")
SDL_LDFLAGS := $(shell pkg-config --libs sdl2 2>/dev/null || echo "-L/opt/homebrew/lib -L/usr/local/lib -lSDL2")

gui: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -o $(BIN_DIR)/game_gui apps/gui/dama_gui.c $(LIB_SRCS) $(LDFLAGS) $(SDL_LDFLAGS)

# Tests
tests: $(BIN_DIR)
	$(CC) $(CFLAGS) -Itests/unit -o $(BIN_DIR)/run_tests tests/unit/test_main.c $(LIB_SRCS) $(LDFLAGS)

# Run tests (builds if needed)
test: tests
	./$(BIN_DIR)/run_tests

test-engine: tests
	./$(BIN_DIR)/run_tests engine

test-search: tests
	./$(BIN_DIR)/run_tests search

test-neural: tests
	./$(BIN_DIR)/run_tests neural

test-training: tests
	./$(BIN_DIR)/run_tests training

test-common: tests
	./$(BIN_DIR)/run_tests common

# Benchmarks
benchmarks: $(BIN_DIR)
	$(CC) $(CFLAGS) -Itests/benchmark -o $(BIN_DIR)/run_bench tests/benchmark/bench_main.c $(LIB_SRCS) $(LDFLAGS)

bench: benchmarks
	./$(BIN_DIR)/run_bench

bench-engine: benchmarks
	./$(BIN_DIR)/run_bench engine

bench-neural: benchmarks
	./$(BIN_DIR)/run_bench neural

bench-mcts: benchmarks
	./$(BIN_DIR)/run_bench mcts

bench-training: benchmarks
	./$(BIN_DIR)/run_bench training

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

.PHONY: all clean gui tests test test-engine test-search test-neural test-training test-common benchmarks bench bench-engine bench-neural bench-mcts bench-training

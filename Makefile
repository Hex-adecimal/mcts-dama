UNAME_S := $(shell uname -s)

CC = gcc-15
CFLAGS = -Wall -Wextra -std=c99 -O3 -march=native -flto -MMD -MP -Isrc
LDFLAGS = -lm

# OpenMP (Linux & macOS with GCC)
ifeq ($(UNAME_S),Linux)
    CFLAGS  += -fopenmp
    LDFLAGS += -fopenmp
endif

ifeq ($(UNAME_S),Darwin)
    # Requires Homebrew GCC (not Apple clang)
    CFLAGS  += -fopenmp
    LDFLAGS += -fopenmp
endif

OBJ_DIR = obj
BIN_DIR = bin
TARGET  = $(BIN_DIR)/dama

# Source files
SRCS = main.c src/game.c src/mcts.c src/debug.c

# Object files (maintains directory structure in obj)
OBJS = $(SRCS:%.c=$(OBJ_DIR)/%.o)
DEPS = $(OBJS:.o=.d)

# Test configuration
TEST_SRCS = test/test_game.c src/game.c src/debug.c
TEST_TARGET = bin/run_tests

# Tournament target
TOURNAMENT_SRCS = tools/tournament.c src/game.c src/mcts.c src/debug.c
TOURNAMENT_TARGET = bin/tournament

all: $(BIN_DIR) $(OBJ_DIR) $(TARGET)

# Convenience alias for user habit (come ti permetti O.O)
main: all
.PHONY: main

tournament: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) -MF $(OBJ_DIR)/tournament.d -o $(TOURNAMENT_TARGET) $(TOURNAMENT_SRCS) -lm $(LDFLAGS)

tuner: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) -MF $(OBJ_DIR)/tuner.d -o bin/tuner tools/tuner.c src/game.c src/mcts.c src/debug.c -lm $(LDFLAGS)

fast: $(BIN_DIR) $(OBJ_DIR)
	$(CC) $(CFLAGS) -MF $(OBJ_DIR)/fast.d -o bin/fast tools/fast_tournament.c src/game.c src/mcts.c src/debug.c -lm $(LDFLAGS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Tests target
tests: $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(TEST_TARGET) $(TEST_SRCS) $(LDFLAGS)

# Pattern rule for object files
# Uses $(dir $@) to create necessary subdirectories in obj/
$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

-include $(DEPS)

.PHONY: all clean
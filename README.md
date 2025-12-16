# MCTS Dama (Italian Checkers)

Monte Carlo Tree Search implementation for Italian Checkers (Dama) with optimized bitboard representation.

## Features

- **Bitboard Engine**: Efficient 64-bit bitboard representation for fast move generation
- **MCTS AI**: Monte Carlo Tree Search with UCB1 selection and smart rollout heuristics
- **Italian Rules**: Full implementation of Italian Checkers rules including mandatory captures and priority rules
- **Tournament Mode**: AI vs AI benchmarking with ELO rating system
- **Performance Optimized**: Aggressive compiler optimizations (-O3, -march=native, -flto)

## Build

```bash
make          # Build main game
make tests    # Build and run unit tests
make tournament # Build tournament mode
make clean    # Clean build artifacts
```

## Usage

### Play a game
```bash
./bin/dama
```

### Run tournament
```bash
./bin/tournament
```

### Run tests
```bash
./bin/run_tests
```

## Configuration

Edit `src/params.h` to tune MCTS parameters:
- `UCB1_C`: Exploration constant (default: 1.414)
- `TIME_WHITE/BLACK`: Time per move in seconds (default: 0.5)
- `WEIGHT_*`: Heuristic weights for rollout policy
- `DEFAULT_USE_LOOKAHEAD`: Enable 1-ply lookahead in endgame (0/1)

## Project Structure

```
├── src/
│   ├── game.c/h       # Core game logic and move generation
│   ├── mcts.c/h       # Monte Carlo Tree Search implementation
│   └── params.h       # Configuration parameters
├── test/
│   └── test_game.c    # Unit tests
├── main.c             # Main game loop
├── main_tournament.c  # Tournament mode
└── Makefile
```

## License

MIT

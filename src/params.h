// Pesi euristica esplorazione albero
#define UCB1_C 1.414
#define WIN_SCORE 1.0
#define DRAW_SCORE 0.25 // tunare questo
#define LOSS_SCORE 0.0

// Pesi euristica mosse
#define WEIGHT_PROMOTION 100
#define WEIGHT_SAFE_EDGE 20
#define WEIGHT_ADVANCE   5
#define WEIGHT_BASE_BREAK 50 // Penalty for moving base pieces too early
#define WEIGHT_DANGER     200 // Penalty for moves that leave pieces exposed to capture

// Configurazione spazio-tempo
#define ARENA_SIZE ((size_t)2048*2 * 1024 * 1024)
#define ANYTIME 0.2
#define TIME_WHITE 0.5
#define TIME_BLACK 0.5 


#define DEFAULT_ROLLOUT_EPSILON 0.2
#define EXPANSION_THRESHOLD 0
#define DEFAULT_USE_LOOKAHEAD 1 // Enable 1-ply lookahead by default
#define DEFAULT_TREE_REUSE 1    // Enable tree reuse by default


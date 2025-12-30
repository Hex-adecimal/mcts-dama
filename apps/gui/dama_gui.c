#include <SDL2/SDL.h>
#include <SDL2/SDL_thread.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "game.h"
#include "movegen.h"
#include "mcts.h"
#include "cnn.h"
#include "mcts.h"
#include "params.h"


// --- CONSTANTS ---
#define WINDOW_WIDTH  800
#define WINDOW_HEIGHT 800
#define BOARD_SIZE    8
#define SQUARE_SIZE   (WINDOW_WIDTH / BOARD_SIZE)
#define PIECE_RADIUS  (SQUARE_SIZE / 2 - 10)

// Colors
const SDL_Color COLOR_LIGHT = {240, 217, 181, 255};
const SDL_Color COLOR_DARK  = {181, 136, 99, 255};
const SDL_Color COLOR_WHITE_PIECE = {255, 255, 240, 255};
const SDL_Color COLOR_BLACK_PIECE = {40, 40, 40, 255};
const SDL_Color COLOR_HINT = {100, 200, 100, 150}; // Green transparent
const SDL_Color COLOR_SELECTED = {255, 255, 0, 150}; // Yellow
const SDL_Color COLOR_LAST_MOVE = {0, 0, 255, 100}; // Blue transparent

const SDL_Color COLOR_ADVISOR = {255, 165, 0, 150}; // Orange transparent

// --- GLOBALS ---
GameState state;
MoveList legal_moves;
MCTSConfig config_gm;
MCTSConfig config_vanilla;
MCTSConfig config_cnn; // Neural Network
MCTSConfig config_advisor; // Advisor uses GM settings
MCTSConfig active_config; // The one we will use
Arena mcts_arena;
Arena advisor_arena; // Separate arena for Advisor thread

CNNWeights cnn_weights; // Shared weights

Node *root = NULL; // Global root for persistence
int human_color = WHITE;
int selected_sq = -1;
int last_move_from = -1;
int last_move_to = -1;

// Advisor Globals
SDL_Thread *advisor_thread = NULL;
bool advisor_running = false;
int advisor_suggest_from = -1;
int advisor_suggest_to = -1;
uint64_t advisor_analyzed_hash = 0; // To validate suggestion relevance

bool is_ai_thinking = false;

// --- HELPER FUNCTIONS ---

// --- HELPER FUNCTIONS ---

void render_board(SDL_Renderer *renderer);
void handle_click(int x, int y);
void ai_move();

// --- ADVISOR LOGIC ---

int advisor_runner(void *data) {
    (void)data; // Unused
    advisor_running = true;
    
    // Copy state closely to avoid race conditions roughly (assuming main thread idle)
    // Critical: Main thread modifies 'state' in apply_move. 
    GameState local_state = state; 
    advisor_analyzed_hash = local_state.hash;

    // Run MCTS (GM)
    arena_reset(&advisor_arena);
    Node *adv_root = mcts_create_root(local_state, &advisor_arena, config_advisor);
    Move adv_move = mcts_search(adv_root, &advisor_arena, TIME_HIGH, config_advisor, NULL, NULL);

    // Print Advisor Tree Stats
    if (adv_root) {
        printf("\n--- Advisor Tree Stats ---\n");
        printf("Root Visits: %d | Depth: %d | Nodes: %d\n", 
               adv_root->visits, get_tree_depth(adv_root), get_tree_node_count(adv_root));
        print_mcts_stats_sorted(adv_root);
    }

    // Output Result
    if (adv_move.path[0] != 0 || adv_move.length > 0) {
        advisor_suggest_from = adv_move.path[0];
        advisor_suggest_to = (adv_move.length > 0) ? adv_move.path[adv_move.length] : adv_move.path[1];
        
        // Human-readable format (e.g., C3 -> D4)
        char from_file = (advisor_suggest_from % 8) + 'A';
        int from_rank = (advisor_suggest_from / 8) + 1;
        char to_file = (advisor_suggest_to % 8) + 'A';
        int to_rank = (advisor_suggest_to / 8) + 1;
        
        printf("[Advisor] Suggests: %c%d -> %c%d (idx %d -> %d)\n", 
               from_file, from_rank, to_file, to_rank,
               advisor_suggest_from, advisor_suggest_to);
    }

    advisor_running = false;
    return 0;
}

void start_advisor() {
    if (advisor_running || advisor_thread != NULL) return; 
    
    // Reset suggestion for new turn
    advisor_suggest_from = -1;
    advisor_suggest_to = -1;
    
    advisor_thread = SDL_CreateThread(advisor_runner, "AdvisorThread", NULL);
}

void clean_advisor() {
    if (advisor_thread) {
        SDL_WaitThread(advisor_thread, NULL);
        advisor_thread = NULL;
    }
}

// ---------------------

void draw_circle(SDL_Renderer *renderer, int cx, int cy, int radius) {
    for (int w = 0; w < radius * 2; w++) {
        for (int h = 0; h < radius * 2; h++) {
            int dx = radius - w; // horizontal offset
            int dy = radius - h; // vertical offset
            if ((dx*dx + dy*dy) <= (radius * radius)) {
                SDL_RenderDrawPoint(renderer, cx + dx, cy + dy);
            }
        }
    }
}

void draw_crown(SDL_Renderer *renderer, int cx, int cy, int radius) {
    // Draw a small inner circle to represent a queen/lady
    SDL_SetRenderDrawColor(renderer, 255, 215, 0, 255); // Gold
    draw_circle(renderer, cx, cy, radius / 3);
}

void render_board(SDL_Renderer *renderer) {
    // Draw Squares
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            SDL_Rect rect = {c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE};
            // Standard chess coloring
            if ((r + c) % 2 == 0) {
                 SDL_SetRenderDrawColor(renderer, COLOR_LIGHT.r, COLOR_LIGHT.g, COLOR_LIGHT.b, 255);
            } else {
                 SDL_SetRenderDrawColor(renderer, COLOR_DARK.r, COLOR_DARK.g, COLOR_DARK.b, 255);
            }
            SDL_RenderFillRect(renderer, &rect);
            
            // Highlight Last Move
            int true_sq = (7 - r) * 8 + c; // Map visual (r,c) to index 0-63
            if (true_sq == last_move_from || true_sq == last_move_to) {
                SDL_SetRenderDrawColor(renderer, COLOR_LAST_MOVE.r, COLOR_LAST_MOVE.g, COLOR_LAST_MOVE.b, COLOR_LAST_MOVE.a);
                SDL_RenderFillRect(renderer, &rect);
            }
            
            // Highlight Advisor Suggestion (If valid for current state)
            if (advisor_suggest_from != -1 && advisor_analyzed_hash == state.hash) {
                 if (true_sq == advisor_suggest_from || true_sq == advisor_suggest_to) {
                    SDL_SetRenderDrawColor(renderer, COLOR_ADVISOR.r, COLOR_ADVISOR.g, COLOR_ADVISOR.b, COLOR_ADVISOR.a);
                    SDL_RenderFillRect(renderer, &rect);
                 }
            }
            
            // Highlight selected
            if (true_sq == selected_sq) {
                SDL_SetRenderDrawColor(renderer, COLOR_SELECTED.r, COLOR_SELECTED.g, COLOR_SELECTED.b, 150);
                SDL_RenderFillRect(renderer, &rect);
            }

            // Highlight hints (valid moves from selection)
            if (selected_sq != -1) {
                for (int i=0; i<legal_moves.count; i++) {
                    if (legal_moves.moves[i].path[0] == selected_sq) {
                         Move *m = &legal_moves.moves[i];
                         int dest = m->path[ (m->length>0) ? m->length : 1 ];
                         if (dest == true_sq) {
                            SDL_SetRenderDrawColor(renderer, 100, 255, 100, 100);
                            SDL_Rect hint = {c * SQUARE_SIZE + 35, r * SQUARE_SIZE + 35, 30, 30};
                            SDL_RenderFillRect(renderer, &hint);
                         }
                    }
                }
            }
        }
    }

    // Draw Pieces
    for (int i = 0; i < 64; i++) {
        int r_idx = i / 8;
        int c_idx = i % 8;
        int visual_r = 7 - r_idx;
        int visual_c = c_idx;
        
        int cx = visual_c * SQUARE_SIZE + SQUARE_SIZE / 2;
        int cy = visual_r * SQUARE_SIZE + SQUARE_SIZE / 2;

        if (check_bit(state.piece[WHITE][PAWN], i) || check_bit(state.piece[WHITE][LADY], i)) {
            SDL_SetRenderDrawColor(renderer, COLOR_WHITE_PIECE.r, COLOR_WHITE_PIECE.g, COLOR_WHITE_PIECE.b, 255);
            draw_circle(renderer, cx, cy, PIECE_RADIUS);
            if (check_bit(state.piece[WHITE][LADY], i)) draw_crown(renderer, cx, cy, PIECE_RADIUS);
        } else if (check_bit(state.piece[BLACK][PAWN], i) || check_bit(state.piece[BLACK][LADY], i)) {
            SDL_SetRenderDrawColor(renderer, COLOR_BLACK_PIECE.r, COLOR_BLACK_PIECE.g, COLOR_BLACK_PIECE.b, 255);
            draw_circle(renderer, cx, cy, PIECE_RADIUS);
            if (check_bit(state.piece[BLACK][LADY], i)) draw_crown(renderer, cx, cy, PIECE_RADIUS);
        }
    }
}

void ai_move() {
    // Ensure Advisor is stopped/cleaned before AI moves (should be implicit as AI moves on opponent turn, but safety first)
    if (advisor_thread) clean_advisor();

    is_ai_thinking = true;
    printf("AI Thinking (Turn: %d)...\n", state.current_player);
    
    SDL_PumpEvents();

    if (root == NULL) {
        printf("Creating new MCTS root...\n");
        arena_reset(&mcts_arena); 
        root = mcts_create_root(state, &mcts_arena, active_config);
    }
    
    Node *new_root = NULL;

    Move best_move = mcts_search(root, &mcts_arena, TIME_HIGH, active_config, NULL, &new_root);
    
    if (best_move.path[0] == 0 && best_move.path[1] == 0 && best_move.length == 0) {
        printf("AI Resigns (No valid moves).\n");
        is_ai_thinking = false;
        return; 
    }

    if (new_root) {
        root = new_root;
    } else {
        root = NULL;
        arena_reset(&mcts_arena); 
    }
    
    last_move_from = best_move.path[0];
    last_move_to = (best_move.length > 0) ? best_move.path[best_move.length] : best_move.path[1];
    
    apply_move(&state, &best_move);
    generate_moves(&state, &legal_moves);
    
    printf("AI Played Move. New Turn: %d\n", state.current_player);
    is_ai_thinking = false;
    
    SDL_Event e;
    while (SDL_PollEvent(&e)); 
}

void handle_click(int x, int y) {
    if (is_ai_thinking) return; // Strict block
    if ((int)state.current_player != human_color) {
        printf("Not your turn!\n"); 
        return;
    }

    int col = x / SQUARE_SIZE;
    int row = y / SQUARE_SIZE;
    int sq_idx = (7 - row) * 8 + col; // Visual to Logic map
    
    // If clicking on our own piece, select it
    uint64_t my_pieces = get_pieces(&state, human_color);
    if (check_bit(my_pieces, sq_idx)) {
        selected_sq = sq_idx;
        return;
    }
    
    // If clicking on a valid destination for the selected piece, move
    if (selected_sq != -1) {
        for (int i = 0; i < legal_moves.count; i++) {
            Move m = legal_moves.moves[i];
            if (m.path[0] == selected_sq) {
                int dest = m.path[ (m.length>0) ? m.length : 1 ];
                if (dest == sq_idx) {
                    // HUMAN MOVE: Invalidate Advisor
                    if (advisor_thread) {
                         // Thread will finish naturally, but we invalidate hash
                         advisor_analyzed_hash = 0; 
                    }
                
                    apply_move(&state, &m);
                    selected_sq = -1;
                    
                    // SET LAST MOVE
                    last_move_from = m.path[0];
                    last_move_to = (m.length > 0) ? m.path[m.length] : m.path[1];
                    
                    // CRITICAL: Invalidate Tree because Human moved randomly (from AI perspective)
                    root = NULL;
                    arena_reset(&mcts_arena); 
                    
                    generate_moves(&state, &legal_moves);
                    return;
                }
            }
        }
        selected_sq = -1;
    }
}

int main(void) {
    // Init Game
    init_game(&state);
    init_move_tables();
    zobrist_init();
    generate_moves(&state, &legal_moves);
    
    arena_init(&mcts_arena, ARENA_SIZE);

    // 1. Config Grandmaster (Hard)
    config_gm = (MCTSConfig){
        .ucb1_c = UCB1_C,
        .rollout_epsilon = ROLLOUT_EPSILON_RANDOM,
        .use_ucb1_tuned = 1,
        .use_tt = 1,
        .use_solver = 1,
        .use_progressive_bias = 1,
        .bias_constant = DEFAULT_BIAS_CONSTANT,
        .use_fpu = 1,
        .fpu_value = FPU_VALUE,
        .use_decaying_reward = 1,
        .decay_factor = DEFAULT_DECAY_FACTOR,
        .weights = { 
            .w_capture = W_CAPTURE, .w_promotion = W_PROMOTION, .w_advance = W_ADVANCE, 
            .w_center = W_CENTER, .w_edge = W_EDGE, .w_base = W_BASE, 
            .w_threat = W_THREAT, .w_lady_activity = W_LADY_ACTIVITY 
        }
    };

    // 2. Config Vanilla (Simple)
    config_vanilla = (MCTSConfig){
        .ucb1_c = 1.414, // Standard
        .rollout_epsilon = 1.0, // Fully random rollouts
        .use_ucb1_tuned = 0,
        .use_tt = 0,
        .use_solver = 0,
        .use_progressive_bias = 0,
        .use_fpu = 0,
        .use_decaying_reward = 0,
        // Weights ignored in Vanilla random rollout usually, but safe to init
        .weights = {0} 
    };

    // 3. Config Neural Network (AlphaZero)
    cnn_init(&cnn_weights);
    int loaded = cnn_load_weights(&cnn_weights, "out/models/cnn_weights_v3.bin"); // Champion
    if (!loaded) loaded = cnn_load_weights(&cnn_weights, "out/models/run_3h_current.bin"); // Fallback to latest
    if (!loaded) loaded = cnn_load_weights(&cnn_weights, "out/models/cnn_weights_final.bin");
    
    if (loaded) {
        printf("✓ Loaded Neural Network weights (Champion/Current).\n");
        // Upgrade GM to Hybrid (The Champion Config)
        config_gm.cnn_weights = &cnn_weights;
    } else {
        printf("⚠️ Failed to load NN weights. Hybrid mode disabled.\n");
    }

    config_cnn = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO);
    config_cnn.cnn_weights = &cnn_weights;
    config_cnn.max_nodes = 2000;

    // 4. Config Advisor (Upgrade to CNN if available)
    if (loaded) {
        config_advisor = config_cnn;
    } else {
        config_advisor = config_gm;
    }
    config_advisor.verbose = 1;

    // DIFFICULTY SELECTION
    int difficulty = 0;
    while (difficulty < 1 || difficulty > 3) {
        printf("\n=== SELEZIONA DIFFICOLTA' ===\n");
        printf("1. Semplice (Vanilla MCTS)\n");
        printf("2. Difficile (Grandmaster Hybrid - Champion) 🏆\n");
        printf("3. Neural Network (AlphaZero) 🧠\n");
        printf("Scelta (1-3): ");
        if (scanf("%d", &difficulty) != 1) {
            while(getchar() != '\n'); // flush
        }
    }
    
    // Set Window Title based on difficulty
    char title[100];
    if (difficulty == 1) {
        snprintf(title, sizeof(title), "MCTS Dama - vs Vanilla Bot (Easy)");
        active_config = config_vanilla;
        printf("\nModalitá: SEMPLICE (Vanilla)\n");
    } else if (difficulty == 2) {
        snprintf(title, sizeof(title), "MCTS Dama - vs Grandmaster Hybrid (Champion)");
        active_config = config_gm;
        printf("\nModalitá: DIFFICILE (Grandmaster Hybrid)\n");
    } else {
        snprintf(title, sizeof(title), "MCTS Dama - vs AlphaZero (Neural Network)");
        active_config = config_cnn;
        printf("\nModalitá: ALPHAZERO (Neural Network)\n");
    }

    // Init SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL Error: %s\n", SDL_GetError());
        return 1;
    }
    
    // Initialize Advisor Arena
    arena_init(&advisor_arena, ARENA_SIZE);

    SDL_Window *window = SDL_CreateWindow(title, 
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
                                          WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) return 1;

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    
    bool running = true;
    SDL_Event e;
    
    printf("GUIDA:\n");
    printf("- Clicca su pedina BIANCA per selezionare.\n");
    printf("- Clicca su casella valida (verde) per muovere.\n");
    printf("- AI risponderà automaticamente.\n");
    printf("- ADVISOR (Arancio) suggerirà mosse al tuo turno.\n");

    while (running) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                running = false;
            } else if (e.type == SDL_MOUSEBUTTONDOWN) {
                if (e.button.button == SDL_BUTTON_LEFT) {
                    handle_click(e.button.x, e.button.y);
                }
            } else if (e.type == SDL_KEYDOWN) {
                // Debug force AI
                if (e.key.keysym.sym == SDLK_SPACE) {
                     printf("DEBUG: Force AI Trigger\n");
                     ai_move();
                }
            }
        }
        
        // Check Game Over
        if (legal_moves.count == 0) {
             SDL_SetWindowTitle(window, "GAME OVER! No legal moves.");
        } else {
             // AI Turn
             if ((int)state.current_player != human_color && !is_ai_thinking) {
                render_board(renderer);
                SDL_RenderPresent(renderer);
                SDL_Delay(200); 
                ai_move();
             } 
             // Human Turn - Run Advisor
             else if ((int)state.current_player == human_color && !is_ai_thinking) {
                 start_advisor(); // Safe to call repeatedly, it checks running flag
             }
        }

        // Render
        SDL_SetRenderDrawColor(renderer, 50, 50, 50, 255);
        SDL_RenderClear(renderer);
        
        render_board(renderer);
        
        SDL_RenderPresent(renderer);
        SDL_Delay(16); // ~60fps
    }

    // Clean threads
    clean_advisor();

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    arena_free(&mcts_arena);
    arena_free(&advisor_arena);

    return 0;
}

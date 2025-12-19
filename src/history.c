#include "history.h"

#define MAX_MOVES 1024

static double history_value[MAX_MOVES];
static int history_count[MAX_MOVES];

/* CODIFICA MOSSA
   Adattala se move_t ha nomi diversi */
static int move_id(move_t m) {
    return (m.from * 32 + m.to) % MAX_MOVES;
}

void history_init(void) {
    for (int i = 0; i < MAX_MOVES; i++) {
        history_value[i] = 0.0;
        history_count[i] = 0;
    }
}

void history_update(move_t move, double result) {
    int id = move_id(move);
    history_value[id] += result;
    history_count[id]++;
}

double history_term(move_t move, int visits) {
    int id = move_id(move);

    if (history_count[id] == 0)
        return 0.0;

    return HISTORY_WEIGHT *
           (history_value[id] / history_count[id]) /
           (visits + 1);
}

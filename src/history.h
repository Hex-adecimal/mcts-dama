#ifndef HISTORY_H
#define HISTORY_H

#include "game.h"

void history_init(void);
void history_update(Move move, double result);
double history_term(Move move, int visits);

#endif

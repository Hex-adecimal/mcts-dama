#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "../src/game.h"


// gcc ./src/game.c test/test_game.c -o run_tests


// --- HELPERS PER I TEST ---

// Pulisce completamente la scacchiera per creare scenari personalizzati
void clear_board(GameState *state) {
    state->white_pieces = 0;
    state->white_ladies = 0;
    state->black_pieces = 0;
    state->black_ladies = 0;
    state->current_player = WHITE;
    state->moves_without_captures = 0;
}

// Stampa info se un test fallisce
void log_test(const char* name) {
    printf("Esecuzione test: %-40s ... ", name);
}

void pass() {
    printf("[OK]\n");
}

// --- TEST CASES ---

void test_simple_move() {
    log_test("Movimento Semplice Pedina (con ostacolo)");
    GameState s;
    clear_board(&s);

    // Setup: Pedina Bianca in C3 (Indice 18)
    set_bit(&s.white_pieces, C3); 
    
    // Ostacolo: Pedina Bianca in B4 (Indice 25)
    // Questo blocca il movimento C3 -> B4
    // MA attenzione: anche B4 può muoversi (verso A5 e C5)!
    set_bit(&s.white_pieces, B4);

    MoveList list;
    generate_moves(&s, &list);
    print_board(&s);
    print_move_list(&list);
    // Assert: Non possiamo controllare list.count == 1, perché anche B4 muove.
    // Dobbiamo cercare specificamente le mosse che partono da C3.
    
    int c3_moves_found = 0;
    int move_to_d4_found = 0;
    int move_to_b4_found = 0;

    for (int i = 0; i < list.count; i++) {
        // Analizziamo solo le mosse che partono da C3
        if (list.moves[i].path[0] == (uint8_t)C3) {
            c3_moves_found++;
            
            if (list.moves[i].path[1] == (uint8_t)D4) {
                move_to_d4_found = 1;
            }
            if (list.moves[i].path[1] == (uint8_t)B4) {
                move_to_b4_found = 1;
            }
        }
    }

    // 1. C3 deve aver generato ESATTAMENTE 1 mossa (quella libera verso D4)
    assert(c3_moves_found == 1);

    // 2. La mossa trovata deve essere verso D4
    assert(move_to_d4_found == 1);

    // 3. La mossa verso B4 NON deve esistere (perché bloccata dall'amico)
    assert(move_to_b4_found == 0);

    // Controllo extra: se applichiamo la mossa corretta, lo stato cambia bene
    // Cerchiamo la mossa giusta nella lista per applicarla
    Move *correct_move = NULL;
    for(int i=0; i<list.count; i++) {
        if (list.moves[i].path[0] == (uint8_t)C3) {
            correct_move = &list.moves[i];
            break;
        }
    }
    
    apply_move(&s, correct_move);
    assert(check_bit(s.white_pieces, D4) == 1); // Arrivato
    assert(check_bit(s.white_pieces, C3) == 0); // Partito
    assert(s.current_player == BLACK); // Turno cambiato
    
    print_board(&s);

    pass();
}

void test_simple_capture() {
    log_test("Presa Semplice Obbligatoria");
    GameState s;
    clear_board(&s);

    // Setup: Bianco in C3, Nero in D4
    set_bit(&s.white_pieces, C3);
    set_bit(&s.black_pieces, D4);

    MoveList list;
    generate_moves(&s, &list);
    print_board(&s);
    print_move_list(&list);
    // Assert: Deve esserci 1 mossa (la presa è obbligatoria, le mosse semplici vietate)
    assert(list.count == 1);

    Move m = list.moves[0];
    // C3 (18) salta D4 (27) e atterra in E5 (36)
    assert(m.path[0] == C3);
    assert(m.path[1] == E5);
    assert(m.length == 1); // Lunghezza 1 salto
    assert(m.captured_squares[0] == D4);

    // Assert: Rimozione pezzo mangiato
    apply_move(&s, &m);
    assert(check_bit(s.black_pieces, D4) == 0); // Il nero è morto
    assert(check_bit(s.white_pieces, E5) == 1); // Il bianco è atterrato
    
    print_board(&s);

    pass();
}

void test_pawn_cannot_capture_lady() {
    log_test("Regola: Pedina NON mangia Dama");
    GameState s;
    clear_board(&s);

    // Setup: Pedina Bianca C3, DAMA Nera D4
    set_bit(&s.white_pieces, C3);
    set_bit(&s.black_ladies, D4); // Dama!

    MoveList list;
    generate_moves(&s, &list);
    print_board(&s);
    print_move_list(&list);

    // Assert: Non deve generare catture.
    // D4 è occupato, quindi la pedina può andare solo in B4 (se libero)
    // Se B4 fosse occupato, list.count sarebbe 0.
    // Qui B4 è vuoto, quindi deve generare la mossa semplice C3->B4
    
    assert(list.count == 1);
    assert(list.moves[0].length == 0); // Mossa semplice
    assert(list.moves[0].path[1] == B4); // Va a sinistra

    print_board(&s);

    pass();
}

void test_promotion() {
    log_test("Promozione a Dama");
    GameState s;
    clear_board(&s);

    // Setup: Pedina Bianca in G7 (sta per promuovere in H8 o F8)
    set_bit(&s.white_pieces, G7);

    MoveList list;
    generate_moves(&s, &list);
    print_board(&s);
    print_move_list(&list);

    // G7 -> H8 e G7 -> F8
    assert(list.count == 2);

    // Applichiamo G7 -> H8
    Move m;
    // Cerchiamo quella che va in H8
    if (list.moves[0].path[1] == H8) m = list.moves[0];
    else m = list.moves[1];

    apply_move(&s, &m);

    // Assert: Non è più pedina, è dama
    assert(check_bit(s.white_pieces, H8) == 0);
    assert(check_bit(s.white_ladies, H8) == 1);
    
    print_board(&s);

    pass();
}

void test_chain_capture() {
    log_test("Presa Multipla (Catena)");
    GameState s;
    clear_board(&s);

    // Setup: Bianco A1. Neri in B2 e D4.
    // Percorso: A1 -> (mangia B2) -> C3 -> (mangia D4) -> E5
    set_bit(&s.white_pieces, A1);
    set_bit(&s.black_pieces, B2);
    set_bit(&s.black_pieces, D4);

    MoveList list;
    generate_moves(&s, &list);
    print_board(&s);
    print_move_list(&list);


    assert(list.count == 1);
    Move m = list.moves[0];

    // Verifica struttura catena
    assert(m.length == 2); // 2 salti
    assert(m.path[0] == A1);
    assert(m.path[1] == C3); // Atterraggio intermedio
    assert(m.path[2] == E5); // Atterraggio finale
    assert(m.captured_squares[0] == B2);
    assert(m.captured_squares[1] == D4);

    // Verifica applicazione
    apply_move(&s, &m);
    assert(check_bit(s.white_pieces, E5) == 1);
    assert(check_bit(s.black_pieces, B2) == 0);
    assert(check_bit(s.black_pieces, D4) == 0);

    print_board(&s);

    pass();
}

void test_priority_quantity() {
    log_test("Priorità: Quantità (2 > 1)");
    GameState s;
    clear_board(&s);

    // Setup:
    // Bianco in E3.
    // Percorso A (1 presa): Nero in F4. (E3->F4->G5)
    // Percorso B (2 prese): Neri in D4, D6. (E3->D4->C5->D6->E7)
    
    set_bit(&s.white_pieces, E3);
    
    set_bit(&s.black_pieces, F4); // Vittima 1 (dx)
    
    set_bit(&s.black_pieces, D4); // Vittima 2 (sx)
    set_bit(&s.black_pieces, D6); // Vittima 3 (sx sequenza)

    MoveList list;
    generate_moves(&s, &list);
    print_board(&s);
    print_move_list(&list);

    // Assert: Deve restituire SOLO la mossa da 2 prese
    assert(list.count == 1);
    assert(list.moves[0].length == 2);
    assert(list.moves[0].path[2] == E7); // Finale del percorso lungo

    print_board(&s);

    pass();
}

void test_priority_quality_mover() {
    log_test("Priorità: Qualità Pezzo (Dama > Pedina)");
    GameState s;
    clear_board(&s);

    // Setup:
    // Pedina Bianca A1 può mangiare B2.
    // Dama Bianca G1 può mangiare F2.
    // Entrambi mangiano 1 pezzo. Ma Dama ha priorità.

    set_bit(&s.white_pieces, A1);
    set_bit(&s.black_pieces, B2);

    set_bit(&s.white_ladies, G1);
    set_bit(&s.black_pieces, F2);

    MoveList list;
    generate_moves(&s, &list);
    print_board(&s);
    print_move_list(&list);

    assert(list.count == 1);
    assert(list.moves[0].is_lady_move == 1); // Deve muovere la Dama
    assert(list.moves[0].path[0] == G1);

    print_board(&s);

    pass();
}

void test_priority_quality_captured() {
    log_test("Priorità: Qualità Prede (Mangia Dama > Mangia Pedina)");
    GameState s;
    clear_board(&s);

    // Setup: Dama Bianca in E3.
    // A dx: Pedina Nera F4. (Salto in G5)
    // A sx: Dama Nera D4. (Salto in C5)
    // Entrambi sono 1 presa con Dama. Vince chi mangia la Dama.

    set_bit(&s.white_ladies, E3);
    set_bit(&s.black_pieces, F4); // Pedina
    set_bit(&s.black_ladies, D4); // Dama

    MoveList list;
    generate_moves(&s, &list);
    print_board(&s);
    print_move_list(&list);

    assert(list.count == 1);
    assert(list.moves[0].captured_ladies_count == 1);
    assert(list.moves[0].captured_squares[0] == D4); // Ha mangiato a sinistra

    print_board(&s);

    pass();
}

int main() {
    printf("--- INIZIO TEST SUITE DAMA ITALIANA ---\n\n");

    test_simple_move();
    test_simple_capture();
    test_pawn_cannot_capture_lady();
    test_promotion();
    test_chain_capture();
    test_priority_quantity();
    test_priority_quality_mover();
    test_priority_quality_captured();

    printf("\n--- TUTTI I TEST PASSATI CON SUCCESSO ---\n");
    return 0;
}
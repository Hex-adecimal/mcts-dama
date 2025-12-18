#include "../src/game.h"
#include "../src/mcts.h"
// Hack: include .c to test static functions like tt_insert/tt_lookup
#include "../src/mcts.c"

#include <assert.h>
#include <stdio.h>

void test_hash_collision_handling() {
    printf("Testing Hash Collision Handling...\n");
    
    // 1. Setup
    Arena arena;
    arena_init(&arena, 1024 * 1024);
    TranspositionTable *tt = tt_create(1024); // Small size
    MCTSConfig config = {0}; // Dummy

    // 2. Create State A
    GameState stateA;
    init_game(&stateA);
    stateA.hash = 0x12345ULL; // Fake Hash
    
    Move dummyMove = {0};
    Node *nodeA = create_node(NULL, dummyMove, stateA, &arena, config);
    
    // 3. Insert A into TT
    tt_insert(tt, nodeA);
    printf("Inserted Node A with Hash %llx\n", stateA.hash);
    
    // Verify A is found
    Node *foundA = tt_lookup(tt, &stateA);
    assert(foundA == nodeA);
    printf("[PASS] Node A found correctly.\n");

    // 4. Create State B (Different State, Same Hash)
    GameState stateB;
    init_game(&stateB);
    // Modify stateB to be real different
    stateB.current_player = BLACK; // A was WHITE
    stateB.white_pieces = 0xFF;    // distinct content
    stateB.hash = 0x12345ULL;      // COLLISION! Same hash as A
    
    // 5. Lookup B
    // BEFORE FIX: This would match because hash matches.
    // AFTER FIX: This should return NULL because state is different.
    Node *foundB = tt_lookup(tt, &stateB);
    
    if (foundB == NULL) {
        printf("[PASS] Collision handled! Different state with same hash returned NULL.\n");
    } else {
        printf("[FAIL] Collision NOT handled! Returned nodeA for stateB.\n");
        // Print why
        if (foundB == nodeA) printf("Returned Node A erroneously.\n");
        exit(1);
    }

    arena_free(&arena);
    tt_free(tt);
}

int main() {
    zobrist_init();
    test_hash_collision_handling();
    printf("All collision tests passed.\n");
    return 0;
}

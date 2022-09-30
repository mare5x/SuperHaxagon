#pragma once
#include "super_client.h"

struct SuperStruct;

namespace super_ai {
    struct GameState_DAGGER {
        float walls[6][2];  // distance and width
        float player_pos;
        float player_slot;
        float wall_speed;
    };

    struct GameState_DQN {
        float walls[6][2];  // distance and width
        float wall_speed;
        
        float n_slots[3];  // 1-hot encoding, [0] = 1 if 4 slots, ..., [2] = 1 if 6 slots
        float cur_slot[6];  // 1-hot encoding, [i] = 1 if i-th slot taken

        float player_pos;  // player angle \in [0,1]
    };

    // Calculate new action for the AI every N-th frame.
    const int DAGGER_UPDATE_INTERVAL = 2;
    const int DQN_UPDATE_INTERVAL = 2;

    /** Instantly move the player to the best safe slot in the game. */
    void make_move_instant(SuperStruct* super);

    /** Determine the best direction to start moving in. Used to emulate human movement.
	    Returns 1 for counter clockwise movement, -1 for clockwise movement and 0 for no movement. */
    int get_move_heuristic(SuperStruct* super);

    /** Use DAGGER (imitation learning) to get the moving direction. */
    int get_move_dagger(SuperStruct* super, bool learning);

    /* Use Deep Q-learning to get the moving direction. */
    int get_move_dqn(SuperStruct* super, bool learning);
    /* For debugging, dump the game state to a file. */
    void dump_game_state_dqn(SuperStruct* super, int action);

    // Call this when training the agent and it dies.
    void report_death(SuperStruct* super);

    // Client for communicating with the Python process.
    extern super_client::SuperClient* client;

    void init();
    void exit();
}

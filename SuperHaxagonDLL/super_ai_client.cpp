#include "stdafx.h"
#include <cmath>
#include "SuperStruct.h"
#include "super_ai.h"
#include "super_client.h"


using super_ai::GameState_DAGGER;
using super_ai::GameState_DQN;

namespace super_ai {
    super_client::SuperClient* client;
}


void _print_arr(const float* arr, int size)
{
	for (int i = 0; i < size; ++i)
		printf("[%d] %3.4f ", i, arr[i]);
	printf("\n");
}

void _print_arr(const int* arr, int size) 
{ 
	for (int i = 0; i < size; ++i)
		printf("[%d] %4d ", i, arr[i]);
	printf("\n");
}

void _print_state(const GameState_DAGGER& state)
{
	for (int i = 0; i < 6; ++i)
		_print_arr(state.walls[i], 2);
	printf("%f %f %f\n", state.player_pos, state.player_slot, state.wall_speed);
}

template<class T>
T clamp(T val, T min, T max)
{
	if (val >= max) return max;
	if (val <= min) return min;
	return val;
}

// <walls> must be an array of size [6][2] [output]
// <walls>[i][0] -- wall distance
// <walls>[i][1] -- wall width
void process_walls(SuperStruct* super, float walls[6][2])
{
	const size_t _DIST = 0;
	const size_t _WIDTH = 1;

    // Normalize width and height using same units
	const float max_dist = 5432;

	for (int i = 0; i < 6; ++i) {
		walls[i][_DIST] = max_dist;
		walls[i][_WIDTH] = 0;
	}

	for (int i = 0; i < super->walls.size(); ++i) {
		SuperStruct::Wall& wall = super->walls[i];

		int slot = wall.slot;
		int dist = wall.distance;
		int width = wall.width;
        
        // False positives (the player can safely move on such slots)
        if (dist + width < 167)
            continue;

		if (dist < walls[slot][_DIST]) {
			walls[slot][_DIST] = dist;
			walls[slot][_WIDTH] = width;
		}
	}

	// Normalize the values.
	for (int i = 0; i < 6; ++i) {
		walls[i][_DIST] = clamp(walls[i][_DIST] / max_dist, 0.0f, 1.0f);
		walls[i][_WIDTH] = clamp(walls[i][_WIDTH] / max_dist, 0.0f, 1.0f);
	}

    // Keep only as many walls as there are slots
    for (int i = super->get_slots(); i < 6; ++i) {
        walls[i][_DIST] = 0.0f;
        walls[i][_WIDTH] = 0.0f;
    }
}

void get_game_state_dagger(SuperStruct* super, GameState_DAGGER* game_state) 
{
	process_walls(super, game_state->walls);
	game_state->player_pos = super->get_player_rotation() / 360.0f;
	game_state->player_slot = super->get_player_slot() / (float)super->get_slots();
	game_state->wall_speed = super->get_wall_speed_percent();
}

void get_game_state_dqn(SuperStruct* super, GameState_DQN* game_state)
{
    process_walls(super, game_state->walls);
    game_state->wall_speed = super->get_wall_speed_percent();
    
    int n_slots = super->get_slots();
    game_state->n_slots[0] = n_slots == 4;
    game_state->n_slots[1] = n_slots == 5;
    game_state->n_slots[2] = n_slots == 6;

    int player_slot = super->get_player_slot();
    for (int i = 0; i < 6; ++i) {
        game_state->cur_slot[i] = player_slot == i;
    }

    game_state->player_pos = super->get_player_rotation() / 360.0f;
    game_state->spin_direction = super->is_world_moving_clockwise() ? -1 : 1;
}

void super_ai::init()
{
    client = new super_client::SuperClient(super_client::g_context);
    client->init();
}

void super_ai::exit() 
{
    client->close();
    delete client;
    super_client::close();
}

void super_ai::report_death(SuperStruct* super)
{
    client->send_episode_score(super->get_elapsed_time()); 
}

int super_ai::get_move_dagger(SuperStruct* super, bool learning)
{
	if (!super->is_player_alive())
		return 0;

	GameState_DAGGER game_state = {};
	get_game_state_dagger(super, &game_state);

    int action = 0;
	if (learning) {
		int expert_action = get_move_heuristic(super);
        action = client->request_action(game_state, expert_action);
    }
    else {
        action = client->request_action(game_state);
    }

	return action;
}

int super_ai::get_move_dqn(SuperStruct * super, bool learning)
{
    if (!super->is_player_alive())
        return 0;

    GameState_DQN game_state = {};
    get_game_state_dqn(super, &game_state);

    int action = client->request_action(game_state);
    return action;
}

void super_ai::dump_game_state_dqn(SuperStruct * super, int action)
{
    if (!super->is_player_alive())
        return;

    GameState_DQN game_state = {};
    get_game_state_dqn(super, &game_state);

    float* game_state_f = (float*)(&game_state);
    int n_floats = 6 * 2 + 1 + 3 + 6 + 1 + 1;
    FILE* file = fopen("game_state_dqn.txt", "a");
    for (int i = 0; i < n_floats; ++i) {
        fprintf(file, "%f ", game_state_f[i]);
    }
    fprintf(file, "\n%d\n", action);
    fclose(file);
}

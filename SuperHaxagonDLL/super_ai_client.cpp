#include "stdafx.h"
#include <cmath>
#include "SuperStruct.h"
#include "super_ai.h"
#include "super_client.h"


using super_ai::GameState_DAGGER;
using super_ai::GameState_DQN;

namespace {
	// IDEA OUTLINE (behavioral cloning):
	// Over the course of some time, sample training data to be used for training.
	// Sample an equal amount of samples for each possible action taken by the ai
	// (to ensure balanced training).
	// Store a percentage of the sampled training data to be used for evaluating
	// the performance of the ann. (That data is not used for training.)
	// After the training data has been acquired, train the ann and measure the
	// performance. (Train by using mini batches of sampled training data.)
	// Continue training until improvement stops. Then start a new game
	// and repeat the process.

	// It turns out that doing the above outlined process isn't very effective
	// because it only trains on "perfect" data, i.e. it doesn't know how to
	// react when it makes a mistake. To fix this, use Direct Policy Learning via 
	// Interactive demonstrator, which is better in the sense that more training states are
	// gathered by querying the expert player when mistakes are made.

    // To train the ANN, we must feed it the input states and the resulting outputs.  // The inputs and outputs are observed by playing the game. 
    // The ANN is fed as input GameStates and it outputs the probability of going
    // left or right (or nowhere) based on the input in the output neurons.
    // All inputs and outputs in the ANN are normalized [0, 1].

    super_client::SuperClient client(super_client::g_context);

	unsigned long frame_counter = 0;
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

	const float max_dist = 5432;
	const float max_width = 2400;  // I don't actually know, just a guess ...

	for (int i = 0; i < 6; ++i) {
		walls[i][_DIST] = max_dist;
		walls[i][_WIDTH] = 0;
	}

	for (int i = 0; i < super->walls.size(); ++i) {
		SuperStruct::Wall& wall = super->walls[i];

		int slot = wall.slot;
		int dist = wall.distance;
		int width = wall.width;

		if (dist < walls[slot][_DIST]) {
			walls[slot][_DIST] = dist;
			walls[slot][_WIDTH] = width;
		}
	}

	// Normalize the values.
	for (int i = 0; i < 6; ++i) {
		walls[i][_DIST] = clamp(walls[i][_DIST] / max_dist, 0.0f, 1.0f);
		walls[i][_WIDTH] = clamp(walls[i][_WIDTH] / max_width, 0.0f, 1.0f);
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
}

void super_ai::init()
{
    client.init();
}

void super_ai::exit() 
{
    client.close();
    super_client::close();
}

void super_ai::report_death(SuperStruct* super)
{
    client.send_episode_score(frame_counter);
	frame_counter = 0;
}

int super_ai::get_move_dagger(SuperStruct* super, bool learning)
{
	if (!super->is_in_game())
		return 0;

	++frame_counter;

	GameState_DAGGER game_state = {};
	get_game_state_dagger(super, &game_state);

    int action = 0;
	if (learning) {
		int expert_action = get_move_heuristic(super);
        action = client.request_action(game_state, expert_action);
    }
    else {
        action = client.request_action(game_state);
    }

	return action;
}

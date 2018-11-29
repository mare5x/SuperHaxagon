#include "stdafx.h"
#include "super_deep_ai.h"
#include "genann.h"
#include "SuperStruct.h"
#include <cmath>
#include <deque>


namespace {
	struct GameState {
		double walls[6][2];  // near and far wall distance for 6 slots
		double player_pos;
		double player_slot;
		double world_rotation;
		double wall_speed;
	};

	struct ReplayEntry {
		GameState state;
		bool action[2];  // which of the two outputs was chosen as the action to take?
		double output[2];  // the output of the ann using the given state as input
	};

	const size_t ANN_INPUTS = 16;
	const size_t ANN_HIDDEN_LAYERS = 2;
	const size_t ANN_OUTPUTS = 2;
	const double ANN_LEARNING_RATE = 0.2;

	const double AI_GAMMA = 0.9;
	const double AI_NEUTRAL_REWARD = 0.15;  // all the AI has to do is be neutral, i.e. not die
	const double AI_LOSS_REWARD = -0.42;  // penalty when it dies

	const size_t MEM_BATCH_SIZE = 32;

	unsigned long frame_counter = 0;

	genann* ann;

	std::deque<ReplayEntry> replay_memory;
}


void train_ann(std::deque<ReplayEntry>& memory_batch, double reward);
void get_game_state(SuperStruct* super, GameState* game_state);


void _print_arr(const double* arr, int size)
{
	for (int i = 0; i < size; ++i)
		printf("[%d] %3.4f ", i, arr[i]);
	printf("\n");
}

void _print_state(const GameState& state)
{
	for (int i = 0; i < 6; ++i)
		_print_arr(state.walls[i], 2);

	printf("\n%f %f %f %f\n", state.player_pos, state.player_slot, 
		state.world_rotation, state.wall_speed);
}

void _print_mem(const ReplayEntry& mem)
{
	printf("state: \n");
	_print_state(mem.state);
	printf("action: %d %d\n", mem.action[0], mem.action[1]);
	printf("output: %f %f\n", mem.output[0], mem.output[1]);
}


size_t arg_max(const double* arr, size_t size)
{
	size_t max_idx = 0;
	double max_val = -1;
	for (size_t i = 0; i < size; ++i) {
		if (arr[i] > max_val) {
			max_val = arr[i];
			max_idx = i;
		}
	}
	return max_idx;
}

void dqn_ai::init()
{
	ann = genann_init(ANN_INPUTS, 2, ANN_INPUTS, ANN_OUTPUTS);
	printf("Initialized DQN AI\n");
}

void dqn_ai::exit()
{
	genann_free(ann);
}

void dqn_ai::report_death(SuperStruct* super)
{
	// Since the only time the AI receives a reward is when it dies,
	// we only need to know whether the AI has just died. 

	static int train_iteration = 0;
	static char time_str[32];
	printf("Training AI [%d] : [%s] ...\n", ++train_iteration, super->get_elapsed_time(time_str));

	// Try training the agent only when it dies, since 
	// otherwise it doesn't get any reward.
	train_ann(replay_memory, AI_LOSS_REWARD);

	frame_counter = 0;
}

int dqn_ai::get_move_dir(SuperStruct * super, bool learning)
{
	if (!super->is_in_game())
		return 0;

	++frame_counter;

	GameState game_state;
	get_game_state(super, &game_state);

	const double* outputs = genann_run(ann, (const double*)(&game_state));
	size_t action_idx = arg_max(outputs, ANN_OUTPUTS);

	int action = (action_idx == 0 ? -1 : 1);

	// Try training every second frame.
	if (learning && frame_counter % 2 == 0) {
		// epsilon exploration
		// record memory

		ReplayEntry mem;
		mem.state = std::move(game_state);
		mem.output[0] = outputs[0];
		mem.output[1] = outputs[1];
		//memcpy(mem.output, outputs, ANN_OUTPUTS);
		mem.action[0] = 0 == action_idx;
		mem.action[1] = 1 == action_idx;

		replay_memory.push_back(mem);

		// Keep the size of the batch bounded.
		if (replay_memory.size() >= MEM_BATCH_SIZE) {
			replay_memory.pop_front();
		}

		//if (batch_counter >= MEM_BATCH_SIZE) {
		//	train_ann(replay_memory, AI_NEUTRAL_REWARD);
		//}
		
		//_print_mem(mem);
	}

	return action;
}


// To train the ANN, we must feed it the input states and the resulting outputs.
// The inputs and outputs are observed by playing the game. When the ai hits a wall
// and dies, we propagate a negative reward to the last N_FRAMES game states.
// The ANN is fed as input GameStates and it outputs the probability of going
// left or right based on the input in two output neurons.
// All inputs and outputs in the ANN are normalized [0, 1] (0.5 is the mean).
// <results> is an array of two elements: the left and right reward.
void train_ann(std::deque<ReplayEntry>& memory_batch, double reward)
{
	double desired_outputs[MEM_BATCH_SIZE][2];
	size_t size = min(MEM_BATCH_SIZE, memory_batch.size());

	// process results

	for (int i = 0; i < size; ++i) {
		const ReplayEntry& mem = memory_batch[i];
		for (int j = 0; j < 2; ++j) {
			desired_outputs[i][j] = mem.output[j];
			if (mem.action[j]) {
				desired_outputs[i][j] += reward * pow(AI_GAMMA, MEM_BATCH_SIZE - 1 - i);
				desired_outputs[i][j] = max(0.0, desired_outputs[i][j]);
			}
		}
	}

	//printf("train_ann:\n");
	//for (int i = 0; i < size; ++i) {
	//	_print_arr(desired_outputs[i], 2);
	//}

	// the most recent results are at the end of the array
	// pop completed
	for (int i = 0; i < size; ++i) {
		const ReplayEntry& mem = memory_batch.back();
		genann_train(ann, (const double*)(&mem.state), desired_outputs[i], ANN_LEARNING_RATE);
		memory_batch.pop_back();
	}
}


// To decrease the amount of work for the ANN, we process the walls 
// for the GameState so that the walls are split up into 'buckets'
// based on distance.
// <walls> must be an array of size [6][2] [output]
// <walls>[i][0] -- near distance
// <walls>[i][1] -- far distance
void process_walls(SuperStruct* super, double walls[6][2])
{
	const size_t _NEAR = 0;
	const size_t _FAR = 1;

	const int min_dist = 150 - super->get_wall_speed();
	const double max_dist = 5432;

	for (int i = 0; i < 6; ++i)
		walls[i][_NEAR] = min_dist;
	for (int i = 0; i < 6; ++i)
		walls[i][_FAR] = 5432;  // this value mustn't be too large otherwise it can cloud the judgement of moves

	// Fill the near and far arrays.
	// near[i] is the end distance of the closest wall on slot i (and less than a certain threshold value).
	// far[i] is the start distance of the closest wall on slot i that is further than a certain threshold value.
	for (int i = 0; i < super->walls.size(); ++i) {
		SuperStruct::Wall& wall = super->walls[i];

		int slot = wall.slot;
		int near_dist = wall.distance;
		int far_dist = wall.distance + wall.width;

		if (near_dist <= walls[slot][_NEAR] && far_dist > walls[slot][_NEAR])
			walls[slot][_NEAR] = far_dist;
		if (near_dist < walls[slot][_FAR] && near_dist > 142)
			walls[slot][_FAR] = near_dist;
	}

	// Normalize the values.
	for (int i = 0; i < 6; ++i) {
		walls[i][_NEAR] /= max_dist;
		walls[i][_FAR] /= max_dist;
	}
}

void get_game_state(SuperStruct* super, GameState* game_state) 
{
	process_walls(super, game_state->walls);
	game_state->player_pos = super->get_player_rotation() / 360.0;
	game_state->player_slot = super->get_player_slot() / (double)super->get_slots();
	game_state->world_rotation = super->get_world_rotation() / 360.0;
	game_state->wall_speed = super->get_wall_speed_percent();
}
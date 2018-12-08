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
		//double world_rotation;
		double wall_speed;
	};

	struct ReplayEntry {
		GameState state;
		bool action[2];  // which of the two outputs was chosen as the action to take?
		double output[2];  // the output of the ann using the given state as input
	};

	const size_t ANN_INPUTS = 16;
	const size_t ANN_HIDDEN_LAYERS = 1;
	const size_t ANN_HIDDEN_SIZE = 8;
	const size_t ANN_OUTPUTS = 2;
	const double ANN_LEARNING_RATE = 0.1;

	const char* ANN_FPATH = "super_weights.ann";

	const double AI_GAMMA = 0.9;
	const double AI_NEUTRAL_REWARD = 0.005;  // all the AI has to do is be neutral, i.e. not die
	const double AI_LOSS_REWARD = -0.5;  // penalty when it dies

	const size_t NEUTRAL_REWARD_FREQUENCY = 42;  // Give out a reward every 32 (skipped) frames.
	const size_t HISTORY_SIZE = 1024;
	const size_t MEM_BATCH_SIZE = 32;

	unsigned long frame_counter = 0;

	genann* ann;

	std::deque<ReplayEntry> replay_memory;

	ReplayEntry* replay_batch[MEM_BATCH_SIZE];
	double desired_ann_outputs[MEM_BATCH_SIZE][ANN_OUTPUTS];
}


/*
	    <History>  <Death>	
	+---------------+---+
	|				| 	|
	+---------------+---+
*/


void train_ann(ReplayEntry** memory_batch, const double desired_outputs[MEM_BATCH_SIZE][ANN_OUTPUTS], size_t size);
void get_game_state(SuperStruct* super, GameState* game_state);
void process_neutral_results(ReplayEntry** memory_batch, double desired_outputs[MEM_BATCH_SIZE][ANN_OUTPUTS], double reward);
void process_death_results(ReplayEntry** memory_batch, double desired_outputs[MEM_BATCH_SIZE][ANN_OUTPUTS], double reward);
void sample_memory(std::deque<ReplayEntry>& memory_batch, ReplayEntry** dst, size_t size);


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

	//printf("\n%f %f %f %f\n", state.player_pos, state.player_slot, 
	//	state.world_rotation, state.wall_speed);
	printf("\n%f %f %f\n", state.player_pos, state.player_slot, state.wall_speed);
}

void _print_mem(const ReplayEntry& mem)
{
	printf("state: \n");
	_print_state(mem.state);
	printf("action: %d %d\n", mem.action[0], mem.action[1]);
	printf("output: %f %f\n", mem.output[0], mem.output[1]);
}


template<class T>
T clamp(T val, T min, T max)
{
	if (val >= max) return max;
	if (val <= min) return min;
	return val;
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

void dqn_ai::init(bool load)
{
	FILE* f;
	if (load && (f = fopen(ANN_FPATH, "r"))) {
		printf("Loading ANN from file [%s] ...\n", ANN_FPATH);
		ann = genann_read(f);
		fclose(f);
	} else {
		ann	= genann_init(ANN_INPUTS, ANN_HIDDEN_LAYERS, ANN_HIDDEN_SIZE, ANN_OUTPUTS);
	}

	printf("Initialized DQN AI [%d] [%d] [%d] [%d]\n", 
			ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);
}

void dqn_ai::exit(bool save)
{
	if (save) {
		FILE* f = fopen(ANN_FPATH, "w");
		genann_write(ann, f);
		fclose(f);
	}

	genann_free(ann);
}

void dqn_ai::report_death(SuperStruct* super)
{
	static int train_iteration = 0;
	static char time_str[32];
	printf("Training AI [%d] : [%s] ...\n", ++train_iteration, super->get_elapsed_time(time_str));

	for (int i = max(replay_memory.size() - MEM_BATCH_SIZE, 0), j = 0; i < replay_memory.size(); ++i, ++j)
		replay_batch[j] = &replay_memory[i];
	process_death_results(replay_batch, desired_ann_outputs, AI_LOSS_REWARD);
	train_ann(replay_batch, desired_ann_outputs, MEM_BATCH_SIZE);

	replay_memory.clear();

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

		ReplayEntry mem;
		mem.state = std::move(game_state);
		mem.output[0] = outputs[0];
		mem.output[1] = outputs[1];
		//memcpy(mem.output, outputs, ANN_OUTPUTS);
		mem.action[0] = 0 == action_idx;
		mem.action[1] = 1 == action_idx;

		replay_memory.push_back(mem);

		// Keep the size of the batch bounded.
		// Q: Should we keep the most recent history or should it be random?
		if (replay_memory.size() > HISTORY_SIZE) {
			replay_memory.pop_front();
		}

		// Give out a survival reward.
		if (frame_counter >= NEUTRAL_REWARD_FREQUENCY && replay_memory.size() > 2 * MEM_BATCH_SIZE) {
			frame_counter = 0;
			sample_memory(replay_memory, replay_batch, MEM_BATCH_SIZE);
			process_neutral_results(replay_batch, desired_ann_outputs, AI_NEUTRAL_REWARD);
			printf("Giving out neutral reward ...\n");
			train_ann(replay_batch, desired_ann_outputs, MEM_BATCH_SIZE);
		}
		
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
void train_ann(ReplayEntry** memory_batch, const double desired_outputs[MEM_BATCH_SIZE][ANN_OUTPUTS], size_t size)
{
	//printf("train_ann:\n");
	//for (int i = 0; i < size; ++i) {
	//	_print_arr(desired_outputs[i], ANN_OUTPUTS);
	//}

	for (int i = 0; i < size; ++i) {
		const ReplayEntry& mem = *memory_batch[i];
		genann_train(ann, (const double*)(&mem.state), desired_outputs[i], ANN_LEARNING_RATE);
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
		walls[i][_FAR] = 5432;

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
	//game_state->world_rotation = super->get_world_rotation() / 360.0;
	game_state->wall_speed = super->get_wall_speed_percent();
}

void process_neutral_results(ReplayEntry** memory_batch, double desired_outputs[MEM_BATCH_SIZE][ANN_OUTPUTS], double reward)
{
	for (int i = 0; i < MEM_BATCH_SIZE; ++i) {
		const ReplayEntry& mem = *memory_batch[i];
		for (int j = 0; j < ANN_OUTPUTS; ++j) {
			desired_outputs[i][j] = mem.output[j];
			if (mem.action[j]) {
				desired_outputs[i][j] += reward;
				desired_outputs[i][j] = clamp(desired_outputs[i][j], 0.0, 1.0);
			}
		}
	}
}

void process_death_results(ReplayEntry** memory_batch, double desired_outputs[MEM_BATCH_SIZE][ANN_OUTPUTS], double reward)
{
	for (int i = 0; i < MEM_BATCH_SIZE; ++i) {
		const ReplayEntry& mem = *memory_batch[i];
		for (int j = 0; j < ANN_OUTPUTS; ++j) {
			desired_outputs[i][j] = mem.output[j];
			if (mem.action[j]) {
				desired_outputs[i][j] += reward * pow(AI_GAMMA, MEM_BATCH_SIZE - 1 - i);
				desired_outputs[i][j] = clamp(desired_outputs[i][j], 0.0, 1.0);
			}
		}
	}
}


/* Randomly sample the <memory_batch> to get <size> elements into <dst>. */
void sample_memory(std::deque<ReplayEntry>& memory_batch, ReplayEntry** dst, size_t size)
{
	// Reservoir-sampling [https://en.wikipedia.org/wiki/Reservoir_sampling]

	size = min(size, memory_batch.size());
	for (int i = 0; i < size; ++i)
		dst[i] = &memory_batch[i];

	for (int i = size; i < memory_batch.size(); ++i) {
		int j = rand() % i;
		if (j < size) {
			dst[j] = &memory_batch[i];
		}
	}
}
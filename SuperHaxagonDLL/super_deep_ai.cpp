#include "stdafx.h"
#include "super_deep_ai.h"
#include "SuperStruct.h"
#include "super_ai.h"
#include "genann.h"
#include <cmath>
#include <deque>
#include <vector>


namespace {
	struct GameState {
		double walls[6][2];  // distance and width
		double player_pos;
		double player_slot;
		double wall_speed;
	};

	struct ReplayEntry {
		GameState state;
		int action;  // which action was taken?
	};

	const size_t ANN_INPUTS = sizeof(GameState) / sizeof(double);
	const size_t ANN_HIDDEN_LAYERS = 2;
	const size_t ANN_HIDDEN_SIZE = 10;
	const size_t ANN_OUTPUTS = 3;
	const double ANN_LEARNING_RATE = 0.1;

	const char* ANN_FPATH = "super_weights.ann";

	const size_t HISTORY_SIZE = 18000;  // 5 minutes of 60 frames per second
	const size_t VALIDATION_SIZE = 0.1 * HISTORY_SIZE;  // How many replays should be stored to be used to evaluate the ann's performance?
	const size_t MEM_BATCH_SIZE = 32;
	const size_t AI_OBSERVATION_STEPS = 8;  // How many steps to observe before training?

	unsigned long frame_counter = 0;

	genann* ann;

	std::deque<ReplayEntry> replay_memory;

	std::vector<ReplayEntry> validation_replays;

	ReplayEntry* replay_batch[MEM_BATCH_SIZE];
	double desired_ann_outputs[MEM_BATCH_SIZE][ANN_OUTPUTS];
}


void train_ann(ReplayEntry** memory_batch, const double desired_outputs[MEM_BATCH_SIZE][ANN_OUTPUTS], size_t size);
void get_game_state(SuperStruct* super, GameState* game_state);
void process_results(ReplayEntry** memory_batch, double desired_outputs[MEM_BATCH_SIZE][ANN_OUTPUTS]);
void sample_memory(std::deque<ReplayEntry>& memory_batch, ReplayEntry** dst, size_t size);
void store_replay(GameState& game_state, int action, std::deque<ReplayEntry>& replay_memory);
void evaluate_performance();


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

	printf("\n%f %f %f\n", state.player_pos, state.player_slot, state.wall_speed);
}

void _print_mem(const ReplayEntry& mem)
{
	printf("state: \n");
	_print_state(mem.state);
	printf("action: %d\n", mem.action);
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

size_t get_action_idx(int action)
{
	switch (action) {
	case -1: return 0;
	case  1: return 1;
	case  0: return 2;
	}
}

int get_action(size_t action_idx)
{
	switch (action_idx) {
	case 0: return 2;
	case 1: return 1;
	case 2: return 0;
	}
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

	validation_replays.reserve(VALIDATION_SIZE);
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
	evaluate_performance();

	replay_memory.clear();
	validation_replays.clear();

	frame_counter = 0;
}

int get_ann_action(const GameState& game_state)
{
	const double* outputs = genann_run(ann, (const double*)&game_state);
	//_print_arr(outputs, ANN_OUTPUTS);
	size_t action_idx = arg_max(outputs, ANN_OUTPUTS);
	return get_action(action_idx);
}

int dqn_ai::get_move_dir(SuperStruct * super, bool learning)
{
	if (!super->is_in_game())
		return 0;

	++frame_counter;

	// Supervised learning idea outline:
	// Use the already functional super_ai to decide which action to take.
	// Observe how the already written ai plays the game and sample
	// the observations to train the ann.

	GameState game_state;
	get_game_state(super, &game_state);

	int action = 0;
	if (learning) {
		// Observe the game state and super_ai's action.

		action = get_move_dir(super);

		store_replay(game_state, action, replay_memory);

		if (frame_counter >= AI_OBSERVATION_STEPS && replay_memory.size() > 2 * MEM_BATCH_SIZE) {
			frame_counter = 0;
			sample_memory(replay_memory, replay_batch, MEM_BATCH_SIZE);
			process_results(replay_batch, desired_ann_outputs);
			train_ann(replay_batch, desired_ann_outputs, MEM_BATCH_SIZE);
		}
	} else {
		action = get_ann_action(game_state);
	}

	return action;
}


// To train the ANN, we must feed it the input states and the resulting outputs.
// The inputs and outputs are observed by playing the game. 
// The ANN is fed as input GameStates and it outputs the probability of going
// left or right (or nowhere) based on the input in the output neurons.
// All inputs and outputs in the ANN are normalized [0, 1].
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
// <walls>[i][0] -- wall distance
// <walls>[i][1] -- wall width
void process_walls(SuperStruct* super, double walls[6][2])
{
	const size_t _DIST = 0;
	const size_t _WIDTH = 1;

	const double max_dist = 5432;
	const double max_width = 2400;  // I don't actually know, just a guess ...

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
		walls[i][_DIST] = clamp(walls[i][_DIST] / max_dist, 0.0, 1.0);
		walls[i][_WIDTH] = clamp(walls[i][_WIDTH] / max_width, 0.0, 1.0);
	}
}

void get_game_state(SuperStruct* super, GameState* game_state) 
{
	process_walls(super, game_state->walls);
	game_state->player_pos = super->get_player_rotation() / 360.0;
	game_state->player_slot = super->get_player_slot() / (double)super->get_slots();
	game_state->wall_speed = super->get_wall_speed_percent();
}

void process_results(ReplayEntry** memory_batch, double desired_outputs[MEM_BATCH_SIZE][ANN_OUTPUTS])
{
	for (int i = 0; i < MEM_BATCH_SIZE; ++i) {
		const ReplayEntry& mem = *memory_batch[i];
		size_t action_idx = get_action_idx(mem.action);
		for (int j = 0; j < ANN_OUTPUTS; ++j) {
			desired_outputs[i][j] = (action_idx == j) * (mem.action == 0 ? 0.5 : 1);
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
		if (j < (memory_batch[i].action == 0 ? 2.0 / 3.0 * size : size))
			dst[j] = &memory_batch[i];
	}
}

void store_replay(GameState& game_state, int action, std::deque<ReplayEntry>& replay_memory)
{
	ReplayEntry mem;
	mem.state = std::move(game_state);
	mem.action = action;

	// Decide whether to store the replay into the replay memory or the validation batch.
	// (reservoir sampling style)
	size_t validation_replays_size = max(4, validation_replays.size());
	if (validation_replays_size < VALIDATION_SIZE) {
		size_t history_size = replay_memory.size();
		int k = rand() % (history_size ? history_size : 4);
		if (k < validation_replays_size) {
			validation_replays.push_back(mem);
			return;
		}
	}

	replay_memory.push_back(mem);

	// Keep the size of the batch bounded.
	// Q: Should we keep the most recent history or should it be random?
	if (replay_memory.size() > HISTORY_SIZE)
		replay_memory.pop_front();
}

void evaluate_performance()
{
	size_t correct_answers = 0;
	for (const ReplayEntry& mem : validation_replays) {
		int ann_action = get_ann_action(mem.state);
		if (ann_action == mem.action)
			++correct_answers;
	}
	size_t incorrect_answers = validation_replays.size() - correct_answers;
	printf("ANN performance: %d/%d correct [%.2f]\n", correct_answers, incorrect_answers, (double)correct_answers / incorrect_answers);
}

#include "stdafx.h"
#include "super_deep_ai.h"
#include "SuperStruct.h"
#include "super_ai.h"
#include "win_console.h"
#include "ProgressBar.h"
#include "genann.h"
#include <cmath>
#include <vector>
#include <array>


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
	const size_t ANN_HIDDEN_SIZE = ANN_INPUTS;
	const size_t ANN_OUTPUTS = 3;
	const double ANN_LEARNING_RATE = 0.001;

	const char* ANN_FPATH = "super_weights.ann";
	const char* TRAINING_DATA_PATH = "training_data.txt";
	const char* VALIDATION_DATA_PATH = "validation_data.txt";

    const size_t DATA_POINTS_PER_ITERATION = 3000;  // Gather this many data points each training iteration.
	const double VALIDATION_DATA_PERCENT = 0.2; 
	const size_t MEM_BATCH_SIZE = 32;

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

	genann* ann;

	std::vector<ReplayEntry> training_data;
	std::array<int, ANN_OUTPUTS> training_data_amounts;  // helper arrays to store the amount of each action currently stored
	std::vector<ReplayEntry> validation_data;
	std::array<int, ANN_OUTPUTS> validation_data_amounts;

    unsigned long data_in_iteration = 0;

	unsigned long frame_counter = 0;
    unsigned long train_iteration = 0;

    double best_performance = 0;
}


void _print_arr(const double* arr, int size)
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

void _print_state(const GameState& state)
{
	for (int i = 0; i < 6; ++i)
		_print_arr(state.walls[i], 2);
	printf("%f %f %f\n", state.player_pos, state.player_slot, state.wall_speed);
}

void _print_mem(const ReplayEntry& mem)
{
	printf("state: \n");
	_print_state(mem.state);
	printf("action: %d\n", mem.action);
}


template<class T>
void sample_array(T* src, size_t src_size, T** dst, size_t dst_size)
{
	// Reservoir-sampling [https://en.wikipedia.org/wiki/Reservoir_sampling]

	size_t size = min(src_size, dst_size);
	for (int i = 0; i < size; ++i)
		dst[i] = &src[i];

	for (int i = size; i < src_size; ++i) {
		int j = rand() % i;
		if (j < size)
			dst[j] = &src[i];
	}
}

template<class T>
T clamp(T val, T min, T max)
{
	if (val >= max) return max;
	if (val <= min) return min;
	return val;
}

template<class T>
size_t arg_max(const T* arr, size_t size)
{
	size_t max_idx = 0;
	for (size_t i = 0; i < size; ++i) {
		if (arr[i] > arr[max_idx])
			max_idx = i;
	}
	return max_idx;
}

template<class T>
size_t arg_min(const T* arr, size_t size) 
{	
	size_t min_idx = 0;
	for (size_t i = 0; i < size; ++i) {
		if (arr[i] < arr[min_idx])
			min_idx = i;
	}
	return min_idx;
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
	case 0: return -1;
	case 1: return 1;
	case 2: return 0;
	}
}

ReplayEntry read_replay_entry(FILE* f)
{
	ReplayEntry mem = {};
	for (int j = 0; j < 6; ++j)
		for (int k = 0; k < 2; ++k)
			fscanf(f, "%lf", &mem.state.walls[j][k]);
	fscanf(f, "%lf %lf %lf", &mem.state.player_pos, &mem.state.player_slot, &mem.state.wall_speed);
	fscanf(f, "%d", &mem.action);
	return mem;
}

void write_replay_entry(FILE* f, const ReplayEntry& mem)
{
	const GameState& state = mem.state;
	for (int j = 0; j < 6; ++j) {
		for (int k = 0; k < 2; ++k)
			fprintf(f, "%f ", state.walls[j][k]);
		fprintf(f, "\n");
	}
	fprintf(f, "%f %f %f\n", state.player_pos, state.player_slot, state.wall_speed);
	fprintf(f, "%d\n", mem.action);
}

void write_replay_data(const char* path, const ReplayEntry* src, size_t size)
{
	FILE* f = fopen(path, "w");
	fprintf(f, "%d\n", size);
	for (int i = 0; i < size; ++i) {
		const ReplayEntry& mem = src[i];
		write_replay_entry(f, mem);
	}
	fclose(f);
}

bool read_replay_data(const char* path, std::vector<ReplayEntry>& dst_data, std::array<int, ANN_OUTPUTS>& dst_amounts)
{
	FILE* f = fopen(path, "r");
	if (!f) return false;

	size_t data_size = 0;
	fscanf(f, "%d", &data_size);
	for (int i = 0; i < data_size; ++i) {
		ReplayEntry mem = read_replay_entry(f);
		dst_data.push_back(mem);
		++dst_amounts[get_action_idx(mem.action)];
	}
	fclose(f);
	return true;
}

void write_results()
{
	printf("Writing results ...\n");

	write_replay_data(TRAINING_DATA_PATH, training_data.data(), training_data.size());
	write_replay_data(VALIDATION_DATA_PATH, validation_data.data(), validation_data.size());

	FILE* f = fopen(ANN_FPATH, "w");
	genann_write(ann, f);
	fclose(f);
}

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

void process_desired_outputs(ReplayEntry** memory_batch, double desired_outputs[][ANN_OUTPUTS], size_t batch_size)
{
	for (int i = 0; i < batch_size; ++i) {
		const ReplayEntry& mem = *memory_batch[i];
		size_t action_idx = get_action_idx(mem.action);
		for (int j = 0; j < ANN_OUTPUTS; ++j)
			desired_outputs[i][j] = action_idx == j;
	}
}

// Decide whether to store the replay into the training or validation set (or skip).
bool store_replay(GameState& game_state, int action)
{
    const double SAMPLE_PROBABILITY = 0.2;

	double k = rand() / (double)RAND_MAX;
    if (k < SAMPLE_PROBABILITY) return false;
    k = (k - SAMPLE_PROBABILITY) / (1 - SAMPLE_PROBABILITY);

	ReplayEntry mem = {};
	mem.state = std::move(game_state);
	mem.action = action;

    if (k < VALIDATION_DATA_PERCENT) {
        validation_data.push_back(mem);
    }
    else {
        training_data.push_back(mem);
    }
    return true;
}

int get_ann_action(const GameState& game_state)
{
	const double* outputs = genann_run(ann, (const double*)&game_state);
	//_print_arr(outputs, ANN_OUTPUTS);
	size_t action_idx = arg_max(outputs, ANN_OUTPUTS);
	return get_action(action_idx);
}

double evaluate_performance(const ReplayEntry* testing_data, size_t size)
{
	size_t correct_answers = 0;
	for (int i = 0; i < size; ++i) {
		const ReplayEntry& mem = testing_data[i];
		int ann_action = get_ann_action(mem.state);
		//printf("%d | %d\n", mem.action, ann_action);
		if (ann_action == mem.action)
			++correct_answers;
	}

	double results = (double)correct_answers / size;
	//printf("ANN performance: %d/%d correct [%.2f]\n", correct_answers, size, results);
	return results;
}

// To train the ANN, we must feed it the input states and the resulting outputs.
// The inputs and outputs are observed by playing the game. 
// The ANN is fed as input GameStates and it outputs the probability of going
// left or right (or nowhere) based on the input in the output neurons.
// All inputs and outputs in the ANN are normalized [0, 1].
void train_batch(ReplayEntry** memory_batch, const double desired_outputs[MEM_BATCH_SIZE][ANN_OUTPUTS], size_t size)
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

void train_ann()
{
	printf("Training iteration: %d ...\n", train_iteration);
    printf("Training samples: %d  ||  Validation samples: %d\n", training_data.size(), validation_data.size());

	//static ReplayEntry* replay_batch[MEM_BATCH_SIZE];
	//static double desired_ann_outputs[MEM_BATCH_SIZE][ANN_OUTPUTS];

	//double prev_perf = -1;
	//double cur_perf = 0;
	//int fail_counter = 0;
	//while (cur_perf - prev_perf > 0.001 || fail_counter < 10) {
	//	//sample_array(training_data.data(), training_data.size(), replay_batch, MEM_BATCH_SIZE);
	//	//process_desired_outputs(replay_batch, desired_ann_outputs, MEM_BATCH_SIZE);
	//	//train_batch(replay_batch, desired_ann_outputs, MEM_BATCH_SIZE);

	//	for (const ReplayEntry& mem : training_data) {
	//		double outputs[ANN_OUTPUTS] = {};
	//		outputs[get_action_idx(mem.action)] = 1;
	//		genann_train(ann, (const double*)(&mem.state), outputs, ANN_LEARNING_RATE);
	//	}

	//	prev_perf = cur_perf;
	//	cur_perf = evaluate_performance(validation_data.data(), validation_data.size());

	//	if (cur_perf - prev_perf <= 0.001)
	//		++fail_counter;
	//	else
	//		fail_counter = 0;
	//}

	//static const size_t ITERATIONS = 5000;
	//for (int iter = 0; iter < ITERATIONS; ++iter) {
	//	sample_array(training_data.data(), training_data.size(), replay_batch, MEM_BATCH_SIZE);
	//	process_desired_outputs(replay_batch, desired_ann_outputs, MEM_BATCH_SIZE);
	//	train_batch(replay_batch, desired_ann_outputs, MEM_BATCH_SIZE);

	//	//for (const ReplayEntry& mem : training_data) {
	//	//	//double outputs[ANN_OUTPUTS] = {};
	//	//	//outputs[get_action_idx(mem.action)] = 1;
	//	//	//genann_train(ann, (const double*)(&mem.state), outputs, ANN_LEARNING_RATE);
	//	//}

	//	if (iter % 100 == 0) {
	//		double perf = evaluate_performance(validation_data.data(), validation_data.size());
	//		printf("Iteration [%d]: %f\n", iter, perf);
	//	}
	//}
	static const size_t EPOCHS = 300;
	static ProgressBar progress_bar(EPOCHS);
	progress_bar.reset();

	for (int it = 0; it < EPOCHS; ++it) {
		for (const ReplayEntry& mem : training_data) {
			double outputs[ANN_OUTPUTS] = {};
			outputs[get_action_idx(mem.action)] = 1;
			genann_train(ann, (const double*)(&mem.state), outputs, ANN_LEARNING_RATE);
		}
        // printf("validation: %.3f\n", evaluate_performance(validation_data.data(), validation_data.size()));
        // printf("training: %.3f\n", evaluate_performance(training_data.data(), training_data.size()));
		progress_bar.update();
	}
	progress_bar.clear();
}

void dqn_ai::init(bool load)
{
	FILE* f;
	if (load && (f = fopen(ANN_FPATH, "r"))) {
		printf("Loading ANN from file [%s] ...\n", ANN_FPATH);
		ann = genann_read(f);
		fclose(f);
	}
	else {
		ann = genann_init(ANN_INPUTS, ANN_HIDDEN_LAYERS, ANN_HIDDEN_SIZE, ANN_OUTPUTS);
	}

	printf("Initialized DQN AI [%d] [%d] [%d] [%d]\n",
		ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);

	if (read_replay_data(TRAINING_DATA_PATH, training_data, training_data_amounts))
		printf("Loading training data ...\n");
	if (read_replay_data(VALIDATION_DATA_PATH, validation_data, validation_data_amounts))
		printf("Loading validation data ...\n");

    // train_ann();
}

void dqn_ai::exit(bool save)
{
	if (save)
		write_results();

	genann_free(ann);
}

void dqn_ai::report_death(SuperStruct* super)
{
    if (data_in_iteration >= DATA_POINTS_PER_ITERATION) {
        train_ann();
        write_results();

        double perf = evaluate_performance(validation_data.data(), validation_data.size());
        if (perf > best_performance) {
            best_performance = perf;
        }

        set_text_formatting(32);
        printf("Iteration [%d]: %f\n", train_iteration, perf);
        set_text_formatting(0);

        ++train_iteration;
        data_in_iteration = 0;
    }
	frame_counter = 0;
}

int dqn_ai::get_move_dir(SuperStruct* super, bool learning)
{
	if (!super->is_in_game())
		return 0;

	++frame_counter;

	GameState game_state = {};
	get_game_state(super, &game_state);

    int action = 0;
	if (learning) {
		int expert_action = get_move_dir(super);

        if (store_replay(game_state, expert_action)) {
            ++data_in_iteration;
            if (data_in_iteration % 100 == 0)
                printf("Iteration [%d]: data acquired %d/%d\n", train_iteration, data_in_iteration, DATA_POINTS_PER_ITERATION);
        }
        
        if (train_iteration == 0)
            action = expert_action;
        else
            action = get_ann_action(game_state);
    }
    else {
        action = get_ann_action(game_state);
    }

	return action;
}
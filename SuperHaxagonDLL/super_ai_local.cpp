#include "super_ai.h"
#include "SuperStruct.h"
#include <cmath>
#include <cstdio>


namespace {
	int slots;
	int world_direction;  // world spinning rotation direction 
	int hedge;
	int near_dst[6];
	int far_dst[6];
}


/* dir should be 1 for ccw and -1 for cw movement. */
inline int get_next_slot(int i, int dir)
{
	return (i + slots + dir) % slots;
}


/** Assigns a score to the movement if the player were to move from slot start_slot 
	to slot end_slot in the given direction. The higher the score, the better. 
	
	Idea thanks to https://github.com/ecx86/superhexagon-internal.*/
int evaluate_move(int start_slot, int end_slot, int dir)
{
	// Prioritize movement in the direction the world is spinning in, because
	// the player moves faster if moving in the same direction as the world.
	const int step_penalty = world_direction == dir ? 28 : 36; 
	
	int cur_slot = start_slot;
	int start_dist = far_dst[cur_slot];
	int step_counter = 0;
	// Penalty values were found by trial and error. Mostly magic.
	int penalty = 0;

	//if (cur_slot == end_slot && far_dst[cur_slot] - near_dst[cur_slot] < hedge) {
	//	penalty += pow(near_dst[cur_slot] + hedge - far_dst[cur_slot], 1.5);
	//}

	while (cur_slot != end_slot) {
		int next_slot = get_next_slot(cur_slot, dir);

		// An obstruction on the next slot.
		//if (near_dst[next_slot] > near_dst[cur_slot]) {
		//	penalty += (near_dst[next_slot] - near_dst[cur_slot]) * 4;
		//}

		// The next slot is unreachable due to an obstruction.
		if (near_dst[next_slot] > far_dst[cur_slot]) {
			penalty += pow(near_dst[next_slot] - far_dst[cur_slot], 1.5);
		}

		// The amount of empty space between the current slot and the next slot is too little,
		// so we most likely can't fit through.
		if (near_dst[cur_slot] > far_dst[next_slot] - hedge) { // || abs(far_dst[cur_slot] - near_dst[next_slot]) < 420) {
			// The exponent value is very important in controlling whether the 
			// player should move through this slot or if it's better to maybe 
			// stay put and wait for the wall to disappear ...
			penalty += pow(near_dst[cur_slot] + hedge - far_dst[next_slot], 1.42); // *pow(0.85, step_counter++);
		}

		penalty += step_penalty;

		cur_slot = next_slot;
	}
	return far_dst[end_slot] - start_dist - penalty;
}


int super_ai::get_move_heuristic(SuperStruct* super)
{
	if (!super->is_player_alive())
		return 0;

	slots = super->get_slots();
	world_direction = super->is_world_moving_clockwise() ? -1 : 1;
	const int min_dist = 150 - super->get_wall_speed();
	hedge = super->get_wall_speed_percent() * 420.0f + 42;

	for (int i = 0; i < 6; ++i)
		near_dst[i] = min_dist;
	for (int i = 0; i < 6; ++i)
		far_dst[i] = 5432;  // this value mustn't be too large otherwise it can cloud the judgement of moves

	// Fill the near_dst and far arrays.
	// near_dst[i] is the end distance of the closest wall on slot i (and less than a certain threshold value).
	// far_dst[i] is the start distance of the closest wall on slot i that is further than a certain threshold value.
	for (int i = 0; i < super->walls.size(); ++i) {
		SuperStruct::Wall& wall = super->walls[i];
		
		int slot = wall.slot;
		int near_dst_dist = wall.distance;
		int far_dist = wall.distance + wall.width;

		if (near_dst_dist <= near_dst[slot] && far_dist > near_dst[slot])
			near_dst[slot] = far_dist;
		if (near_dst_dist < far_dst[slot] && near_dst_dist > 142)
			far_dst[slot] = near_dst_dist;
	}

	// Find the best slot and direction based on the score evaluation algorithm.
	int current_slot = super->get_player_slot();
	int best_slot = current_slot;
	int best_dir = 0;
	int max_score = 0;
	static int prev_dir = 0;

	for (int target_slot = 0; target_slot < slots; ++target_slot) {
		for (int dir = -1; dir <= 1; ++dir) {
			if (dir == 0) continue;

			int score = evaluate_move(current_slot, target_slot, dir);
			// Add a penalty when changing direction to get rid of sudden direction changing
			// during inappropriate times.
			int direction_changed_penalty = (dir == prev_dir ? 0 : 42);
			if (score > max_score + direction_changed_penalty) {
				max_score = score;
				best_dir = target_slot == current_slot ? 0 : dir;
				best_slot = target_slot;
			}
		}
	}

	// If the best slot is the current slot, center the player.
	if (best_slot == current_slot) {
		float slot = super->get_player_rotation() * slots / 360.0f;
		// Elegant way to get the offset of the player from the center of a slot.
		float dir = fmodf(slot, 1.0f) - 0.5f;
		if (dir > 0.1f)
			best_dir = -1;
		else if (dir < -0.1f)
			best_dir = 1;
	}

/*
	printf("---------\n");
	for (int i = 0; i < 6; ++i)
		printf("%6d", near_dst[i]);
	printf("\n");
	for (int i = 0; i < 6; ++i)
		printf("%6d", far_dst[i]);
	printf("\n");

	printf("Current %d to new %d in direction %d based on score %d\n", current_slot, best_slot, best_dir, max_score);
*/

	prev_dir = best_dir;

	return best_dir;
}

void super_ai::make_move_instant(SuperStruct * super)
{
	if (!super->is_player_alive())
		return;

	// Find the closest walls for each slot. Then move to the slot, whose closest wall is the farthest away.

	slots = super->get_slots();

	int closest_walls[6] = {};
	for (int i = 0; i < slots; ++i) closest_walls[i] = 0xffff;

	for (int i = 0; i < super->walls.size(); ++i) {
		SuperStruct::Wall& wall = super->walls[i];
		if (wall.distance < closest_walls[wall.slot])
			closest_walls[wall.slot] = wall.distance;
	}

	DWORD max_dist = 0;
	DWORD best_slot = 0;
	for (int i = 0; i < slots; ++i) {
		if (closest_walls[i] > max_dist) {
			max_dist = closest_walls[i];
			best_slot = i;
		}
	}
	super->set_player_slot(best_slot);
}

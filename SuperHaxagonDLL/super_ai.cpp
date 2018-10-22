#include "super_ai.h"
#include "SuperStruct.h"
#include <cmath>
#include <cstdio>


int slots;
int world_direction;  // world spinning rotation direction 
int near[6];
int far[6];


/* dir should be 1 for ccw and -1 for cw movement. */
inline int next_slot(int i, int dir)
{
	return (i + slots + dir) % slots;
}


/** Assigns a score to the movement if the player were to move from slot i 
	to slot n in the given direction. The higher the score, the better. 
	
	Idea thanks to https://github.com/ecx86/superhexagon-internal.*/
int evaluate_move(int i, int n, int dir)
{
	// Prioritize movement in the direction the world is spinning in, because
	// the player moves faster if moving in the same direction as the world.
	const int step_penalty = world_direction == dir ? 42 : 69; 
	
	int start_dist = far[i];
	int penalty = 0;

	while (i != n) {
		int next_i = next_slot(i, dir);

		// An obstruction on the next slot.
		if (near[next_i] > near[i]) {
			penalty += (near[next_i] - near[i]) * 4;
		}

		// The next slot is unreachable due to an obstruction.
		if (near[next_i] > far[i]) {
			penalty += (near[next_i] - far[i]) * 8;
		}

		// The amount of empty space between the current slot and the next slot is too little,
		// so we most likely can't fit through.
		if (near[i] > far[next_i] - 420) { // || abs(far[i] - near[next_i]) < 420) {
			penalty += 4 * (near[i] + 420 - far[next_i]);
		}

		penalty += step_penalty;

		i = next_i;
	}
	return far[n] - start_dist - penalty;
}


int get_move_dir(SuperStruct* super)
{
	if (!super->is_in_game())
		return 0;

	slots = super->get_slots();
	world_direction = super->is_world_moving_clockwise() ? -1 : 1;

	for (int i = 0; i < 6; ++i)
		near[i] = 142;
	for (int i = 0; i < 6; ++i)
		far[i] = 1 << 13;

	// Fill the near and far arrays.
	// near[i] is the end distance of the closest wall on slot i (and less than a certain threshold value).
	// far[i] is the start distance of the closest wall on slot i that is further than a certain threshold value.
	for (int i = 0; i < super->walls.size(); ++i) {
		SuperStruct::Wall& wall = super->walls[i];
		
		int slot = wall.slot;
		int near_dist = wall.distance;
		int far_dist = wall.distance + wall.width;

		if (near_dist <= near[slot] && far_dist > near[slot])
			near[slot] = far_dist;
		if (near_dist < far[slot] && near_dist > 142)
			far[slot] = near_dist;
	}

	// Find the best slot and direction based on the score evaluation algorithm.
	int current_slot = super->get_player_slot();
	int best_slot = current_slot;
	int best_dir = 0;
	int max_score = 0;
	
	for (int target_slot = 0; target_slot < slots; ++target_slot) {
		for (int dir = -1; dir <= 1; ++dir) {
			if (dir == 0) continue;

			int score = evaluate_move(current_slot, target_slot, dir);
			if (score > max_score) {
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

	printf("---------\n");
	for (int i = 0; i < 6; ++i)
		printf("%6d", near[i]);
	printf("\n");
	for (int i = 0; i < 6; ++i)
		printf("%6d", far[i]);
	printf("\n");

	printf("Current %d to new %d in direction %d based on score %d\n", current_slot, best_slot, best_dir, max_score);

	return best_dir;
}

void autoplay_instant(SuperStruct * super)
{
	if (!super->is_in_game())
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

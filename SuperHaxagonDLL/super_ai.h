#pragma once

struct SuperStruct;

/** Determine the best direction to start moving in. Used to emulate human movement.
	Returns 1 for counter clockwise movement, 1 for clockwise movement and 0 for no movement. */
int get_move_dir(SuperStruct* super);

/** Instantly move the player to the best safe slot in the game. */
void autoplay_instant(SuperStruct* super);
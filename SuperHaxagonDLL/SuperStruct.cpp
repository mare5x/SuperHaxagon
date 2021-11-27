#include "stdafx.h"
#include "SuperStruct.h"
#include "memory_tools.h"


void SuperStruct::Wall::print() const { printf("%d %d %d %d %d\n", slot, distance, width, other, enabled); }

void SuperStruct::update()
{
    update_walls();

    // Shift array and store time
    for (int i = 0; i < prev_times.size() - 1; ++i) {
        prev_times[i + 1] = prev_times[i];
    }
    prev_times[0] = get_elapsed_time();
}

bool SuperStruct::is_fullscreen() const { return read_memory<bool>(base_adr + 0x24); }

bool SuperStruct::is_player_alive() const
{
    // is_in_game is true for a bit of time even after the player dies in-game.
    if (!is_in_game()) return false;

    // Not in-game if all times are equal (the in-game clock is stopped).
    for (int i = 0; i < prev_times.size() - 1; ++i) {
        if (prev_times[i] != prev_times[i + 1]) {
            return true;
        }
    }
    return false;
}

char* SuperStruct::get_elapsed_time(char * const dest_string) const
{
	int time = get_elapsed_time();
	sprintf(dest_string, "%d:%2d", time / 60, time % 60);
	return dest_string;
}

bool SuperStruct::is_player_centered() const
{
	float slot = get_player_rotation() * get_slots() / 360.0f;
	return abs(fmodf(slot, 1.0f) - 0.5f) < 0.1f;

	//int angle = get_player_rotation();
	//int n = get_slots();
	//for (int k = 0; k < n; ++k)
	//	if (abs(angle - ((360 * k + 180) / n)) < 4)
	//		return true;
	//return false;
}

int SuperStruct::slot_to_world_angle(int slot) const
{
	int sides = get_slots();
	float alpha = 360.0f / sides;
	int offset = 0;
	// Why does this even work?
	if (sides == 4) offset = 45;
	else if (sides == 5) offset = 60;
	return alpha * (slot - (sides == 6 ? 1 : 0)) - offset + get_world_rotation();
}

DWORD SuperStruct::get_base_address()
{
	// Since ASLR (Address Space Load Randomization) is off for Super Hexagon, we can use static addresses. 
	return read_memory<DWORD>(get_proc_address() + 0x2857F0);
}

void SuperStruct::update_walls()
{
	// Walls are written to an array by increasing the index each time, which wraps around when full.
	//int n_walls = read_offset<int>(N_WALLS);
	// not actually 64, but who cares -> read everything and parse only valid results
	DWORD wall_adr = base_adr + WALL_START;
	read_memory<Wall>(wall_adr, walls.items, 64 * sizeof(Wall));

	int n_walls = 0;
	for (int i = 0; i < 64; ++i) {
		Wall& wall = walls[i];
		if (wall.is_valid(get_slots())) {
			walls[n_walls] = wall;
			++n_walls;
		}
	}
	walls.set_size(n_walls);
}

template<class T>
inline T SuperStruct::read_offset(OFFSETS offset) const { return read_memory<T>(base_adr + offset); }

template<class T>
inline void SuperStruct::write_offset(OFFSETS offset, const T val) const { write_memory<T>(base_adr + offset, val); }


template DWORD SuperStruct::read_offset(OFFSETS offset) const;
template int SuperStruct::read_offset(OFFSETS offset) const;
template bool SuperStruct::read_offset(OFFSETS offset) const;

template void SuperStruct::write_offset(OFFSETS offset, const int val) const;
template void SuperStruct::write_offset(OFFSETS offset, const DWORD val) const;
template void SuperStruct::write_offset(OFFSETS offset, const bool val) const;

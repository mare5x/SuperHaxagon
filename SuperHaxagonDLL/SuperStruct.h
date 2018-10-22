#pragma once
#include "VariableArray.h"

typedef unsigned long DWORD;

struct SuperStruct {
	// Specifies how an in-game wall/obstacle is laid out in memory. (Thanks to github.com/zku)
	struct Wall {
		DWORD slot;
		DWORD distance;
		DWORD width;
		DWORD other;
		DWORD enabled;

		void print() const;
		bool is_valid(int slots) const { return enabled && distance >= 0 && width > 0 && slot >= 0 && slot < slots; }
	};

	enum WORLD_ROTATION_OPTIONS : DWORD {
		CW_SLOW = 0,
		CCW_SLOW = 1,
		CW_MEDIUM = 2,
		CCW_MEDIUM = 3,
		CW_FAST = 4,
		CCW_FAST = 5,
		CW_VERY_FAST = 6,
		CCW_VERY_FAST = 7,
		SPECIAL = 8
	};

	SuperStruct() : base_adr(get_base_address()), walls(64) { }

	void update() { update_walls(); }

	bool is_fullscreen() const;
	bool is_in_game() const { return read_offset<bool>(IN_GAME); }

	int get_mouse_x() const { return read_offset<int>(MOUSE_X); }
	int get_mouse_y() const { return read_offset<int>(MOUSE_Y); }

	int get_world_rotation() const { return read_offset<int>(WORLD_ROTATION); }
	int get_polygon_radius() const { return read_offset<int>(POLYGON_RADIUS); }
	int get_slots() const { return read_offset<int>(POLYGON_SIDES); }
	

	void set_world_rotation_type(WORLD_ROTATION_OPTIONS type) { write_offset<DWORD>(WORLD_ROTATION_TYPE, type); }
	WORLD_ROTATION_OPTIONS get_world_rotation_type() const { return static_cast<WORLD_ROTATION_OPTIONS>(read_offset<DWORD>(WORLD_ROTATION_TYPE)); }

	bool is_world_moving_clockwise() const
	{
		return get_world_rotation_type() % 2 == 0;
	}

	int get_slot_center_angle(int slot) const
	{
		return 360 / get_slots() * (slot + 0.5);
	}

	void set_player_slot(int slot) const
	{
		DWORD angle = get_slot_center_angle(slot);
		write_offset<DWORD>(PLAYER_ROTATION_1, angle);
		write_offset<DWORD>(PLAYER_ROTATION_2, angle);
	}

	int get_player_slot() const
	{
		return get_player_rotation() * get_slots() / 360.0f;
	}

	int get_player_rotation() const { return read_offset<int>(PLAYER_ROTATION_1); }

	int get_n_walls() const { return read_offset<int>(N_WALLS); }

	bool is_player_centered() const;

	int slot_to_world_angle(int slot) const;

	DWORD base_adr;

	VariableArray<Wall> walls;
private:
	enum OFFSETS : DWORD {
		MOUSE_X = 0x8,
		MOUSE_Y = 0xC,

		WORLD_ROTATION = 0x1AC,
		POLYGON_RADIUS = 0x1B0,
		POLYGON_SIDES = 0x1BC,

		WALL_START = 0x220,  // The start of an array containing Walls.
		N_WALLS = 0x2930,

		PLAYER_ROTATION_1 = 0x2954,
		PLAYER_ROTATION_2 = 0x2958,

		WORLD_ROTATION_TYPE = 0x2968,

		IN_GAME = 0x40668
	};

	/* Returns the base address of the structure that holds most of the interesting properties of the game. */
	static DWORD get_base_address();

	template<class T>
	T read_offset(OFFSETS offset) const;

	template<class T>
	void write_offset(OFFSETS offset, const T val) const;

	void update_walls();
};
#pragma once
#include "memory_tools.h"


namespace SuperHaxagon {
	struct SuperStruct {
		// Specifies how an in-game wall/obstacle is laid out in memory. (Thanks to github.com/zku)
		struct Wall {
			DWORD slot;
			DWORD distance;
			DWORD other[3];
		};

		SuperStruct() : base_adr(get_base_address()) { }

		void update() { update_walls(); }

		bool is_fullscreen() const { return read_memory<bool>(base_adr + 0x24); }

		int get_mouse_x() const { return read_offset<int>(MOUSE_X); }
		int get_mouse_y() const { return read_offset<int>(MOUSE_Y); }

		int get_world_rotation() const { return read_offset<int>(WORLD_ROTATION); }
		int get_polygon_radius() const { return read_offset<int>(POLYGON_RADIUS); }
		int get_polygon_sides() const { return read_offset<int>(POLYGON_SIDES); }

		void set_player_slot(int slot) const
		{
			DWORD angle = 360 / get_polygon_sides() * (slot + 0.5f);
			write_offset<DWORD>(PLAYER_ROTATION_1, angle);
			write_offset<DWORD>(PLAYER_ROTATION_2, angle);
		}

		int get_player_slot() const
		{
			return get_player_rotation() * get_polygon_sides() / 360.0f;
		}

		int get_player_rotation() const { return read_offset<int>(PLAYER_ROTATION_1); }

		int get_n_walls() const { return read_offset<int>(N_WALLS); }

		int slot_to_world_angle(int slot) const
		{
			int sides = get_polygon_sides();
			float alpha = 360.0f / sides;
			int offset = 0;
			// Why does this even work?
			if (sides == 4) offset = 45;
			else if (sides == 5) offset = 60;
			return alpha * (slot - (sides == 6 ? 1 : 0)) - offset + get_world_rotation();
		}

		DWORD base_adr;

		Wall walls[64];  // Max number of walls should be less than 64 ...

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
			PLAYER_ROTATION_2 = 0x2958
		};

		/* Returns the base address of the structure that holds most of the interesting properties of the game. */
		static DWORD get_base_address()
		{
			// Since ASLR (Address Space Load Randomization) is off for Super Hexagon, we can use static addresses. 
			return read_memory<DWORD>(get_proc_address() + 0x2857F0);
		}

		template<class T>
		T read_offset(OFFSETS offset) const { return read_memory<T>(base_adr + offset); }

		template<class T>
		void write_offset(OFFSETS offset, const T val) const { write_memory<T>(base_adr + offset, val); }

		void update_walls()
		{
			// Walls are written to an array by increasing the index each time, which wraps around when full.
			//int n_walls = read_offset<int>(N_WALLS);
			int n_walls = 64;  // not actually 64, but who cares -> read everything and parse only valid results
			DWORD wall_adr = base_adr + WALL_START;
			read_memory<Wall>(wall_adr, walls, n_walls * sizeof(Wall));
		}
	};

	void hook(HMODULE dll);
	void WINAPI unhook();

	void draw();
}
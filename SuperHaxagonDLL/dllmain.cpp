// dllmain.cpp : Defines the entry point for the DLL application.
#include "stdafx.h"
#include <cstdio>
#include <fstream>
#include "memory_tools.h"


const double PI = acos(-1);


typedef void(__stdcall* p_glutSwapBuffers)();
p_glutSwapBuffers orig_glutSwapBuffers;

HMODULE haxagon_dll;
WNDPROC orig_wnd_proc;
BYTE* post_detour_cave;

bool hooked = false;


struct MouseState { int x, y; };
MouseState mouse_state;


struct SuperStruct {
	// Specifies how an in-game wall/obstacle is laid out in memory. (Thanks to github.com/zku)
	struct Wall {
		DWORD slot;
		DWORD distance;
		DWORD other[3];
	};

	SuperStruct() : base_adr(get_base_address()) { }

	void update() { update_walls(); }

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


SuperStruct super;


void draw_text(const char* text, int x, int y)
{
	glRasterPos2d(x, y);
	int len = strlen(text);
	for (int i = 0; i < len; ++i)
		glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, text[i]);
}


void draw_debug_strings()
{
	char text[16] = {};
	snprintf(text, 16, "%d %d", mouse_state.x, mouse_state.y);
	draw_text(text, 5, 28);

	snprintf(text, 16, "%d", super.get_world_rotation());
	draw_text(text, 5, 28 * 2);

	snprintf(text, 16, "%d", super.get_polygon_radius());
	draw_text(text, 5, 28 * 3);

	snprintf(text, 16, "%d", super.get_polygon_sides());
	draw_text(text, 5, 28 * 4);

	snprintf(text, 16, "%d", super.get_player_rotation());
	draw_text(text, 5, 28 * 5);
}


void draw()
{
	/*
	glPushMatrix();
	glTranslated(768/2, 480/2, 0);
	glutWireSphere(250, 50, 50);
	glPopMatrix();
	*/
	
	glBegin(GL_POINTS);
	
	glColor3f(1.0f, 0, 0);
	glVertex2d(mouse_state.x, mouse_state.y);

	glEnd();

	super.update();

	draw_debug_strings();

	glPushMatrix();
	// Make the camera origin be the center of the screen.
	glTranslated(768 / 2, 480 / 2, 0);

	// Find the closest walls for each slot. Then move to the slot, whose closest wall is the farthest away.
	DWORD slots = super.get_polygon_sides();
	DWORD min_dist[6] = {};
	for (int i = 0; i < slots; ++i) min_dist[i] = 0xffff;

	// Draw lines to the closest walls.
	glBegin(GL_LINES);
	for (int i = 0; i < 64; ++i) {
		DWORD& wall_slot = super.walls[i].slot;
		DWORD& wall_dist = super.walls[i].distance;
		
		if (wall_slot < 0 || wall_slot > slots) continue;
		if (wall_dist < min_dist[wall_slot] && wall_dist > 0)
			min_dist[wall_slot] = wall_dist;

		if (wall_dist > 1000 || wall_dist <= 0) continue;

		int deg = super.slot_to_world_angle(wall_slot);
		double rad = -deg * PI / 180;

		glVertex2d(0, 0);
		glVertex2d(cos(rad) * wall_dist, sin(rad) * wall_dist);
	}
	glEnd();

	glPopMatrix();

	DWORD max_dist = 0;
	DWORD best_slot = 0;
	for (int i = 0; i < slots; ++i) {
		if (min_dist[i] > max_dist) {
			max_dist = min_dist[i];
			best_slot = i;
		}
	}
	super.set_player_slot(best_slot);
}


void __stdcall hooked_swap_buffers()
{
	// This function is executed by Super Hexagon's main thread.
	// That means the thread has an OpenGL context and we can gladLoadGL
	// to get access to all gl functions. (A thread can only make OpenGL 
	// calls if it has a rendering context. - https://docs.microsoft.com/en-us/windows/desktop/opengl/rendering-context-functions,
	// and glad makes gl calls).

	if (!hooked) {
		if (gladLoadGL()) {
			printf("OpenGL: %d.%d\n", GLVersion.major, GLVersion.minor);
			glPointSize(4);
		}
		hooked = true;
	}

	draw();
}


void __declspec(naked) trampoline()
{
	__asm {
		PUSHFD
		PUSHAD
		CALL hooked_swap_buffers
		POPAD
		POPFD
		JMP post_detour_cave
	}
}


void WINAPI un_hook(LPVOID hwnd)
{
	remove_detour_hook((DWORD)orig_glutSwapBuffers, post_detour_cave, 9);
	delete[] post_detour_cave;

	glutPassiveMotionFunc(NULL);
	//glutSpecialFunc(NULL);

	glutSetWindowTitle("Super Hexagon");

	_fcloseall();
	FreeConsole();

	FreeLibraryAndExitThread(haxagon_dll, 0);
}


LRESULT CALLBACK input_handler(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	// Un-hook on F1.
	if (uMsg == WM_KEYDOWN && wParam == VK_F1) {
		// Restore the original WNDPROC function.
		SetWindowLongPtr((HWND)hwnd, GWLP_WNDPROC, (LONG_PTR)orig_wnd_proc);
		// The current execution thread is Super Hexagon's MainThread, so exiting this thread would close the program.
		CreateThread(NULL, NULL, (LPTHREAD_START_ROUTINE)&un_hook, &hwnd, NULL, NULL);
		return true;
	} else if (uMsg == WM_KEYDOWN && (wParam == 0x51 || wParam == 0x45)) {
		// Use the Q and E keys to move the player cursor to the next or previous slot.
		super.set_player_slot((super.get_player_slot() + (wParam == 0x51 ? 1 : -1)) % super.get_polygon_sides());
	}

	return orig_wnd_proc(hwnd, uMsg, wParam, lParam);
}


// NOT used.
void glut_special_key_event(int key, int x, int y)
{
	if (key == GLUT_KEY_F1) {
		CreateThread(NULL, NULL, (LPTHREAD_START_ROUTINE)&un_hook, NULL, NULL, NULL);
	}
}


void glut_mouse_move_event(int x, int y)
{
	mouse_state.x = x;
	mouse_state.y = y;
	//printf("%d %d\n", x, y);
}


void WINAPI run_bot(LPVOID param = NULL)
{
	// Create a console shell for debugging purposes.
	AllocConsole();
	FILE* _f;
	freopen_s(&_f, "CONOUT$", "w", stdout);
	printf("Hello, world\n");
	printf("SuperStruct base: %x\n", super.base_adr);

	// Since Super Hexagon uses glut32.dll, we can directly use actual glut methods.
	glutSetWindowTitle("You've been gnomed!");
	HWND hwnd = FindWindow(NULL, L"You've been gnomed!");

	glutCreateMenu(NULL);
	glutAddMenuEntry("Test1", 0);
	glutAddMenuEntry("Test 2222", 1);
	glutAttachMenu(GLUT_MIDDLE_BUTTON);

	// Hook the window input handler.
	orig_wnd_proc = (WNDPROC) GetWindowLongPtr(hwnd, GWLP_WNDPROC);
	SetWindowLongPtr(hwnd, GWLP_WNDPROC, (LONG_PTR)&input_handler);

	// This is easy but it doesn't handle arrow keys used in-game. So, no.
	// glutSpecialFunc(&glut_special_key_event);
	
	// Hook glutSwapBuffers. To get the function's address we can't just use &glutSwapAddress, since
	// that version is a stub in this DLL that points to the actual original function in glut32.dll.
	HMODULE glut_handle = GetModuleHandle(L"glut32.dll");
	orig_glutSwapBuffers = (p_glutSwapBuffers)GetProcAddress(glut_handle, "glutSwapBuffers");
	post_detour_cave = detour_hook((DWORD)orig_glutSwapBuffers, (DWORD)&trampoline, 9);

	glutPassiveMotionFunc(&glut_mouse_move_event);
}


BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
	if (ul_reason_for_call == DLL_PROCESS_ATTACH) {
		haxagon_dll = hModule;
		HANDLE thread = CreateThread(NULL, NULL, (LPTHREAD_START_ROUTINE)&run_bot, NULL, NULL, NULL);
		CloseHandle(thread);
	}
    return TRUE;
}


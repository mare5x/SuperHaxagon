#include "stdafx.h"
#include "SuperHaxagon.h"
#include "glut_hook.h"


enum MENU_OPTION : int {
	DEBUG_STRINGS, AUTOPLAY, CONSOLE, ZOOM
};

const double PI = acos(-1);

int WINDOW_WIDTH = 768;
int WINDOW_HEIGHT = 480;
const char* WINDOW_TITLE = "H4X0R";

// Used to allow window resizing. 
RECT window_rect = { 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT };

int mouse_x, mouse_y;

bool setting_auto_play = true;
bool setting_debug_strings = true;
bool setting_console = true;
bool setting_zoom = false;

HMODULE g_dll;
HWND g_hwnd;

WNDPROC orig_wnd_proc;

SuperHaxagon::SuperStruct super;

DWORD render_adr;			// the address of the hook in the original function
DWORD render_return_adr;	// where to return in the original function
DWORD render_call_adr;		// the new CALL address 
std::array<BYTE, 5> orig_render_bytes;


void open_console();
void close_console();
LRESULT CALLBACK input_handler(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

void __stdcall hooked_render();

void glut_mouse_move_event(int x, int y);
void glut_menu_func(int value);

void hook_glut(const char* title);
void unhook_glut();

void draw_text(const char* text, int x, int y);
void draw_debug_strings();


void open_console()
{
	AllocConsole();
	freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);
}


void close_console()
{
	fclose(stdout);
	FreeConsole();
}


LRESULT CALLBACK input_handler(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	// Un-hook on F1.
	if (uMsg == WM_KEYDOWN && wParam == VK_F1) {
		// Restore the original WNDPROC function.
		SetWindowLongPtr((HWND)hwnd, GWLP_WNDPROC, (LONG_PTR)orig_wnd_proc);
		// The current execution thread is Super Hexagon's MainThread, so exiting this thread would close the program.
		CreateThread(NULL, NULL, (LPTHREAD_START_ROUTINE)&SuperHaxagon::unhook, &hwnd, NULL, NULL);
		return true;
	}
	else if (uMsg == WM_KEYDOWN && (wParam == 0x51 || wParam == 0x45)) {
		// Use the Q and E keys to move the player cursor to the next or previous slot.
		super.set_player_slot((super.get_player_slot() + (wParam == 0x51 ? 1 : -1)) % super.get_polygon_sides());
	}

	// Allow window resizing:
	if (uMsg == WM_SIZING)
		window_rect = *(RECT*)lParam;
	if (uMsg == WM_EXITSIZEMOVE)
		glutReshapeWindow(window_rect.right - window_rect.left, window_rect.bottom - window_rect.top);

	return orig_wnd_proc(hwnd, uMsg, wParam, lParam);
}


void __stdcall hooked_render()
{
	// Executed before any rendering happens in Super Hexagon's main thread.

	// glad must be initialized to use gl functions.
	if (!glut_hook::gl_is_hooked()) {
		glut_hook::init_gl();
	}

	if (setting_zoom) {
		// Zoom out by translating the 'z' coordinate.
		glTranslatef(0, 0, -500);

		// or zoom out by setting the ortho projection matrix.
		//glOrtho(0, 4, 0, 4, 0, 1);
	}
}


void __declspec(naked) render_trampoline()
{
	__asm {
		PUSHFD
		PUSHAD
		CALL hooked_render
		POPAD
		POPFD

		// JMP post_render_cave
		// Original code replaced by hook:
		CALL[render_call_adr]
		// JMP back to original code:
		JMP[render_return_adr]
	}
}


void SuperHaxagon::draw()
{	
	/*
	glPushMatrix();
	glTranslated(768/2, 480/2, 0);
	glutWireSphere(250, 50, 50);
	glPopMatrix();
	*/

	super.update();

	if (setting_debug_strings) {
		glBegin(GL_POINTS);

		glColor3f(1.0f, 0, 0);
		glVertex2d(mouse_x, mouse_y);

		glEnd();

		draw_debug_strings();
	}

	if (!setting_auto_play) return;

	glPushMatrix();
	// Make the camera origin be the center of the screen.
	glTranslated(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2, 0);

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


void SuperHaxagon::hook(HMODULE dll)
{
	// Create a console shell for debugging purposes.
	open_console();
	printf("Hello, world\n");
	printf("SuperStruct base: %x\n", super.base_adr);

	g_dll = dll;
	hook_glut(WINDOW_TITLE);
	g_hwnd = FindWindowA(NULL, WINDOW_TITLE);

	// Hook the window input handler.
	orig_wnd_proc = (WNDPROC)GetWindowLongPtr(g_hwnd, GWLP_WNDPROC);
	SetWindowLongPtr(g_hwnd, GWLP_WNDPROC, (LONG_PTR)&input_handler);

	// Hook Super Hexagon's main render function. The function is called before glSwapBuffers, which
	// allows us to set the GL state before the game draws its own stuff.
	// Special care when hooking, because we are hooking a CALL instruction. That means we can't 
	// just execute that same instruction from somewhere else, since the CALL contains a relative address.
	render_adr = get_proc_address() + 0x75B6D;
	memcpy(orig_render_bytes.data(), (BYTE*)render_adr, orig_render_bytes.size());
	jump_hook(render_adr, (DWORD)&render_trampoline, 5);
	render_call_adr = get_proc_address() + 0x653d0;
	render_return_adr = render_adr + 5;
}


void WINAPI SuperHaxagon::unhook()
{
	unhook_glut();

	write_code_buffer(render_adr, orig_render_bytes.data(), 5);

	fclose(stdout);
	FreeConsole();

	FreeLibraryAndExitThread(g_dll, 0);
}


void glut_mouse_move_event(int x, int y)
{
	mouse_x = x;
	mouse_y = y;
}


void glut_menu_func(int value)
{
	switch (value) {
	case MENU_OPTION::AUTOPLAY:
		setting_auto_play = !setting_auto_play;
		break;
	case MENU_OPTION::DEBUG_STRINGS:
		setting_debug_strings = !setting_debug_strings;
		break;
	case MENU_OPTION::CONSOLE:
		setting_console = !setting_console;
		if (setting_console) open_console();
		else close_console();
		break;
	case MENU_OPTION::ZOOM:
		setting_zoom = !setting_zoom;
		break;
	default:
		break;
	}
}


void hook_glut(const char* title)
{
	glut_hook::hook_SwapBuffers(&SuperHaxagon::draw);

	// Since Super Hexagon uses glut32.dll, we can directly use actual glut methods.
	glutSetWindowTitle(title);

	// This is easy but it doesn't handle arrow keys used in-game. So, no.
	// glutSpecialFunc(&glut_special_key_event);

	glutPassiveMotionFunc(&glut_mouse_move_event);

	// Middle mouse click menu:
	// Activating the menu also works as an in-game pause!
	glutCreateMenu(&glut_menu_func);
	glutAddMenuEntry("Enable/disable autoplay", MENU_OPTION::AUTOPLAY);
	glutAddMenuEntry("Show/hide debug lines", MENU_OPTION::DEBUG_STRINGS);
	glutAddMenuEntry("Open/close debug console", MENU_OPTION::CONSOLE);
	glutAddMenuEntry("Enable/disable zoom out", MENU_OPTION::ZOOM);
	glutAttachMenu(GLUT_MIDDLE_BUTTON);
}


void unhook_glut()
{
	glut_hook::unhook_SwapBuffers();

	glutPassiveMotionFunc(NULL);

	glutSetWindowTitle("Super Hexagon");

	glutReshapeWindow(WINDOW_WIDTH, WINDOW_HEIGHT);
}


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
	snprintf(text, 16, "%d %d", mouse_x, mouse_y);
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
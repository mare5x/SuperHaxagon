#include "stdafx.h"
#include "glut_hook.h"
#include "SuperHaxagon.h"
#include "super_ai.h"
#include "memory_tools.h"
#include "win_console.h"
#include "super_deep_ai.h"
#include "Renderer.h"
#include "BitmapPlusPlus.hpp"


namespace fmodex {
	typedef int(__stdcall *p_FMOD_System_GetWaveData)(
		void* system,
		float* wavearray,
		int numvalues,
		int channeloffset
	);

	HMODULE fmodex_dll;
	p_FMOD_System_GetWaveData FMOD_System_GetWaveData;

	void* _system;

	bool _hooked = false;

	void init(DWORD base_adr)
	{
		fmodex_dll = GetModuleHandle(L"fmodex.dll");
		FMOD_System_GetWaveData = (p_FMOD_System_GetWaveData)GetProcAddress(fmodex_dll, "FMOD_System_GetWaveData");
		_system = read_memory<void*>(base_adr + 0x28da88);  // this is where superhexagon.exe stores the required 'system' parameter

		_hooked = true;
	}

	bool is_hooked()
	{
		return _hooked;
	}
}


enum MENU_OPTION : int {
	DEBUG_LINES, AUTOPLAY, AUTOPLAY_NATURAL, AUTOPLAY_INSTANT, AUTOPLAY_DQN, AI_LEARNING, CONSOLE, ZOOM
};

typedef SuperStruct::WORLD_ROTATION_OPTIONS ROTATION_OPTIONS;

const double PI = acos(-1);

const char* WINDOW_TITLE = "Super Haxagon";
int VIEWPORT_WIDTH = 768;
int VIEWPORT_HEIGHT = 480;

// Used to allow window resizing. 
RECT window_rect = { 0, 0, 768 - 1, 480 - 1};

int mouse_x, mouse_y;
bool console_change_requested = false;

bool setting_autoplay = true;
int setting_autoplay_type = MENU_OPTION::AUTOPLAY_NATURAL;
bool setting_debug_lines = true;
bool setting_console = true;
bool setting_zoom = false;
int setting_rotation_type = -1;
int setting_wall_speed = -1;
bool setting_auto_restart = false;  // automatically restart the game when dead
bool setting_ai_learning = false;

HMODULE g_dll;
HWND g_hwnd;
DWORD g_proc_adr;  // superhexagon.exe address

WNDPROC orig_wnd_proc;

SuperStruct super;

DWORD render_adr;			// the address of the hook in the original function
DWORD render_return_adr;	// where to return in the original function
DWORD render_call_adr;		// the new CALL address 
std::array<BYTE, 5> orig_render_bytes;

int moving_direction = 0;   // 1 (counter clockwise), 0 (not moving), -1 (clockwise)

Renderer renderer;

// Function declarations in this file.
LRESULT CALLBACK input_handler(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

void __stdcall hooked_render();

void glut_mouse_move_event(int x, int y);
void glut_menu_func(int value);
void glut_rotation_speed_menu_func(int option);

void hook_glut(const char* title);
void unhook_glut();

void draw_text(const char* text, int x, int y);
void draw_debug_strings();

void start_moving(int direction);
void stop_moving();

void screenshot(const char* path);


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
		super.set_player_slot((super.get_player_slot() + (wParam == 0x51 ? 1 : -1)) % super.get_slots());
	}
    else if (uMsg == WM_KEYDOWN && wParam == VK_F12) {
        // F12 for screenshot.
        screenshot("screenshot.bmp");
    }

	// Allow window resizing:
	if (uMsg == WM_SIZING)
		window_rect = *(RECT*)lParam;
	if (uMsg == WM_EXITSIZEMOVE)
		glutReshapeWindow(window_rect.right - window_rect.left + 1, window_rect.bottom - window_rect.top + 1);

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

	if (setting_rotation_type != -1)
		super.set_world_rotation_type(static_cast<ROTATION_OPTIONS>(setting_rotation_type));

    if (renderer.shader.ID == -1) {
        renderer.init(); 
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


void draw_debug()
{
	glBegin(GL_POINTS);

	glColor3f(1.0f, 0, 0);
	glVertex2d(mouse_x, mouse_y);

	glEnd();

	draw_debug_strings();

	glPushMatrix();
	// Make the camera origin be the center of the screen.
	glTranslated(VIEWPORT_WIDTH / 2, VIEWPORT_HEIGHT / 2, 0);

	// Draw lines to the closest walls.
	glBegin(GL_LINES);
	for (int i = 0; i < super.walls.size(); ++i) {
		SuperStruct::Wall& wall = super.walls[i];

		if (wall.distance > 1000) continue;

		int deg = super.slot_to_world_angle(wall.slot);
		double rad = -deg * PI / 180;

		glVertex2d(0, 0);
		glVertex2d(cos(rad) * wall.distance, sin(rad) * wall.distance);
	}
	glEnd();

	glPopMatrix();
}


void SuperHaxagon::draw()
{	
   	update();

    renderer.render();  // Render shader stuff

	if (fmodex::is_hooked() && setting_debug_lines) {
		// Draw a pulsing sphere based on the audio data.
		static float arr[1024];
		fmodex::FMOD_System_GetWaveData(fmodex::_system, arr, 1024, 0);
		float avg = 0;
		for (int i = 0; i < 1024; ++i)
			avg += abs(arr[i]) / 1024;
			
		glPushMatrix();
		glTranslated(VIEWPORT_WIDTH / 2, VIEWPORT_HEIGHT / 2, 0);
		glutWireSphere(256 * avg, 16, 16);
		glPopMatrix();
	}

	if (setting_debug_lines) {
		draw_debug();
	}
}


void SuperHaxagon::update()
{
	static bool was_fullscreen = false;
	if (was_fullscreen != super.is_fullscreen()) {
		was_fullscreen = !was_fullscreen;
		if (was_fullscreen) {
			VIEWPORT_WIDTH = glutGet(GLUT_SCREEN_WIDTH);
			VIEWPORT_HEIGHT = glutGet(GLUT_SCREEN_HEIGHT);
		}
		else {
			VIEWPORT_WIDTH = 768;
			VIEWPORT_HEIGHT = 480;
		}
	}

	super.update();

	if (setting_wall_speed != -1)
		super.set_wall_speed(setting_wall_speed);

	static bool in_game = false;
	static bool restart_key_pending = false;

	if (!super.is_in_game()) {
		// Careful! This doesn't necessarily mean the agent died!
		if (in_game) {
			if (setting_autoplay_type == AUTOPLAY_DQN && setting_ai_learning)
				dqn_ai::report_death(&super);
			in_game = false;
			
			if (setting_auto_restart || setting_ai_learning) {
				SendMessage(g_hwnd, WM_KEYDOWN, VK_SPACE, 0);
				restart_key_pending = true;
			}
		}
	}
    else {
		in_game = true;

        if (restart_key_pending) {
            SendMessage(g_hwnd, WM_KEYUP, VK_SPACE, 0);
            restart_key_pending = false;
        }
    }

	if (!setting_autoplay) return;

	switch (setting_autoplay_type) {
	case AUTOPLAY_NATURAL:
		start_moving(get_move_dir(&super));
		break;
	case AUTOPLAY_INSTANT:
		stop_moving();
		autoplay_instant(&super);
		break;
	case AUTOPLAY_DQN:
		start_moving(dqn_ai::get_move_dir(&super, setting_ai_learning));
		break;
	}

	if (console_change_requested) {
		setting_console = !setting_console;
		if (setting_console)
			open_console();
		else
			close_console();
		console_change_requested = false;
	}
}


void SuperHaxagon::hook(HMODULE dll)
{
	// Create a console shell for debugging purposes.
	open_console();
	printf("Hello, world\n");
	printf("SuperStruct base: %x\n", super.base_adr);

	g_dll = dll;
	g_proc_adr = get_proc_address();

	hook_glut(WINDOW_TITLE);
	g_hwnd = FindWindowA(NULL, WINDOW_TITLE);

	// Hook the window input handler.
	orig_wnd_proc = (WNDPROC)GetWindowLongPtr(g_hwnd, GWLP_WNDPROC);
	SetWindowLongPtr(g_hwnd, GWLP_WNDPROC, (LONG_PTR)&input_handler);

	// Hook Super Hexagon's main render function. The function is called before glSwapBuffers, which
	// allows us to set the GL state before the game draws its own stuff.
	// Special care when hooking, because we are hooking a CALL instruction. That means we can't 
	// just execute that same instruction from somewhere else, since the CALL contains a relative address.
	render_adr = g_proc_adr + 0x75B6D;
	memcpy(orig_render_bytes.data(), (BYTE*)render_adr, orig_render_bytes.size());
	jump_hook(render_adr, (DWORD)&render_trampoline, 5);
	render_call_adr = g_proc_adr + 0x653d0;
	render_return_adr = render_adr + 5;

	fmodex::init(g_proc_adr);

	dqn_ai::init();
}


void WINAPI SuperHaxagon::unhook()
{
	dqn_ai::exit();

	stop_moving();
	
	unhook_glut();

	write_code_buffer(render_adr, orig_render_bytes.data(), 5);
	
	close_console();

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
	case MENU_OPTION::DEBUG_LINES:
		setting_debug_lines = !setting_debug_lines;
		break;
	case MENU_OPTION::CONSOLE:
		console_change_requested = true;
		break;
	case MENU_OPTION::ZOOM:
		setting_zoom = !setting_zoom;
		break;
	default:
		break;
	}
}


void glut_rotation_speed_menu_func(int option)
{
	setting_rotation_type = option;
}


void glut_wall_speed_menu_func(int speed)
{
	setting_wall_speed = speed;
}


void glut_autoplay_menu_func(int option)
{
	switch (option) {
	case MENU_OPTION::AUTOPLAY:
		setting_autoplay = !setting_autoplay;
		break;
	case MENU_OPTION::AI_LEARNING:
		setting_ai_learning = !setting_ai_learning;
		break;
	case MENU_OPTION::AUTOPLAY_NATURAL:
		setting_autoplay_type = AUTOPLAY_NATURAL;
		setting_autoplay = true;
		break;
	case MENU_OPTION::AUTOPLAY_INSTANT:
		setting_autoplay_type = AUTOPLAY_INSTANT;
		setting_autoplay = true;
		break;
	case MENU_OPTION::AUTOPLAY_DQN:
		setting_autoplay_type = AUTOPLAY_DQN;
		setting_autoplay = true;
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

	int rotation_speed_menu = glutCreateMenu(&glut_rotation_speed_menu_func);
	glutAddMenuEntry("DEFAULT", -1);
	glutAddMenuEntry("CW_SLOW", ROTATION_OPTIONS::CW_SLOW);
	glutAddMenuEntry("CCW_SLOW", ROTATION_OPTIONS::CCW_SLOW);
	glutAddMenuEntry("CW_MEDIUM", ROTATION_OPTIONS::CW_MEDIUM);
	glutAddMenuEntry("CCW_MEDIUM", ROTATION_OPTIONS::CCW_MEDIUM);
	glutAddMenuEntry("CW_FAST", ROTATION_OPTIONS::CW_FAST);
	glutAddMenuEntry("CCW_FAST", ROTATION_OPTIONS::CCW_FAST);
	glutAddMenuEntry("CW_VERY_FAST", ROTATION_OPTIONS::CW_VERY_FAST);
	glutAddMenuEntry("CCW_VERY_FAST", ROTATION_OPTIONS::CCW_VERY_FAST);
	glutAddMenuEntry("SPECIAL", ROTATION_OPTIONS::SPECIAL);

	int wall_speed_menu = glutCreateMenu(&glut_wall_speed_menu_func);
	glutAddMenuEntry("DEFAULT", -1);
	glutAddMenuEntry("VERY SLOW", 8);
	glutAddMenuEntry("SLOW", 22);
	glutAddMenuEntry("MEDIUM", 33);
	glutAddMenuEntry("FAST", 40);
	glutAddMenuEntry("VERY FAST", 50);

	int autoplay_menu = glutCreateMenu(&glut_autoplay_menu_func);
	glutAddMenuEntry("Enable/disable autoplay", MENU_OPTION::AUTOPLAY);
	glutAddMenuEntry("Enable/disable dqn learning", MENU_OPTION::AI_LEARNING);
	glutAddMenuEntry("Natural movements", MENU_OPTION::AUTOPLAY_NATURAL);
	glutAddMenuEntry("Instant movements", MENU_OPTION::AUTOPLAY_INSTANT);
	glutAddMenuEntry("Deep Reinforcement Learning", MENU_OPTION::AUTOPLAY_DQN);

	glutCreateMenu(&glut_menu_func);
	glutAddSubMenu("Autoplay settings", autoplay_menu);
	glutAddMenuEntry("Show/hide debug lines", MENU_OPTION::DEBUG_LINES);
	glutAddMenuEntry("Open/close debug console", MENU_OPTION::CONSOLE);
	glutAddMenuEntry("Enable/disable zoom out", MENU_OPTION::ZOOM);
	glutAddSubMenu("Set rotation speed", rotation_speed_menu);
	glutAddSubMenu("Set wall speed", wall_speed_menu);
	glutAttachMenu(GLUT_MIDDLE_BUTTON);
}


void unhook_glut()
{
	glut_hook::unhook_SwapBuffers();

	glutPassiveMotionFunc(NULL);

	glutSetWindowTitle("Super Hexagon");

	if (!super.is_fullscreen())
		glutReshapeWindow(glutGet(GLUT_INIT_WINDOW_WIDTH), glutGet(GLUT_INIT_WINDOW_HEIGHT));
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
	const int x = 5;
	const int y = 28;
	int line = 1;

	static char text[1024] = {};
	snprintf(text, sizeof(text), "%d %d", mouse_x, mouse_y);
	draw_text(text, x, y * line++);

	snprintf(text, sizeof(text), "%d", super.get_world_rotation());
	draw_text(text, x, y * line++);

	//snprintf(text, sizeof(text), "%d", super.get_slots());
	//draw_text(text, x, y * line++);

	snprintf(text, sizeof(text), "%d", super.get_player_rotation());
	draw_text(text, x, y * line++);

	snprintf(text, sizeof(text), "%d", super.get_wall_speed());
	draw_text(text, x, y * line++);

    line = 4;
    int w = window_rect.right - window_rect.left + 1;
    int h = window_rect.bottom - window_rect.top + 1;
    snprintf(text, sizeof(text), "Learn: %d", setting_ai_learning);
    draw_text(text, w - 128, h - y * line--);
    snprintf(text, sizeof(text), "AI control: %d", setting_autoplay);
    draw_text(text, w - 128, h - y * line--);
    snprintf(text, sizeof(text), "AI type: %d", setting_autoplay_type);
    draw_text(text, w - 128, h - y * line--);
}

void start_moving(int direction)
{
	if (direction != moving_direction)
		stop_moving();
	
	if (direction != 0)
		SendMessage(g_hwnd, WM_KEYDOWN, direction > 0 ? VK_LEFT : VK_RIGHT, 0);
	
	moving_direction = direction;
}

void stop_moving()
{
	if (moving_direction == 0) return;

	SendMessage(g_hwnd, WM_KEYUP, moving_direction > 0 ? VK_LEFT : VK_RIGHT, 0);
	
	moving_direction = 0;
}

void screenshot(const char* path)
{
    printf("Screenshot: %s\n", path);
    int w = window_rect.right - window_rect.left + 1;
    int h = window_rect.bottom - window_rect.top + 1;
    // NOTE: Using bytes and 3*w*h doesn't work because of 4 byte alignment.
    // Instead use RGBA with 32-bit ints.
    int32_t* pixels = new int32_t[w * h];
    // Read pixels starting from lower left corner:
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    bmp::Bitmap img(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int c = pixels[y * w + x];
            // NOTE: RGBA is stored in memory as ABGR ...
            img.Set(x, h - y - 1, bmp::Pixel((c & 0xFF), (c & 0xFF00) >> 8, (c & 0xFF0000) >> 16));
        }
    }
    img.Save(path);
    delete[] pixels;
}
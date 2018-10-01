// dllmain.cpp : Defines the entry point for the DLL application.
#include "stdafx.h"
#include <cstdio>
#include <fstream>
#include "memory_tools.h"


typedef void(__stdcall* p_glutSwapBuffers)();
p_glutSwapBuffers orig_glutSwapBuffers;

HMODULE haxagon_dll;
WNDPROC orig_wnd_proc;
BYTE* post_detour_cave;

bool hooked = false;


struct MouseState {
	int x, y;
};
MouseState mouse_state;


void draw_text(const char* text)
{
	glRasterPos2d(5, 29);
	int len = strlen(text);
	for (int i = 0; i < len; ++i)
		glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, text[i]);
}


void draw()
{
	glutWireSphere(250, 50, 50);
	
	glBegin(GL_POINTS);
	
	glColor3f(1.0f, 0, 0);
	glVertex2d(mouse_state.x, mouse_state.y);

	glEnd();

	char text[16] = {};
	snprintf(text, 16, "%d %d", mouse_state.x, mouse_state.y);
	draw_text(text);
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
	}

	return orig_wnd_proc(hwnd, uMsg, wParam, lParam);
}


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
	// Since Super Hexagon uses glut32.dll, we can directly use actual glut methods.
	glutSetWindowTitle("You've been gnomed!");
	HWND hwnd = FindWindow(NULL, L"You've been gnomed!");
	
	// Hook the window input handler.
	orig_wnd_proc = (WNDPROC) GetWindowLongPtr(hwnd, GWLP_WNDPROC);
	SetWindowLongPtr(hwnd, GWLP_WNDPROC, (LONG_PTR)&input_handler);

	// This is easy but it doesn't handle arrow keys used in-game. So, no.
	// glutSpecialFunc(&glut_special_key_event);

	// Create a console shell for debugging purposes.
	AllocConsole();
	FILE* _f;
	freopen_s(&_f, "CONOUT$", "w", stdout);
	printf("Hello, world\n");
	
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


// dllmain.cpp : Defines the entry point for the DLL application.
#include "stdafx.h"
#include <cstdio>
#include <fstream>
#include "GL/glut.h"


HMODULE haxagon_dll;
WNDPROC orig_wnd_proc;


void WINAPI un_hook(LPVOID hwnd)
{
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
		// The current execution thread is the MainThread, so exiting this thread would close the program.
		CreateThread(NULL, NULL, (LPTHREAD_START_ROUTINE)&un_hook, &hwnd, NULL, NULL);
		return true;
	}

	return orig_wnd_proc(hwnd, uMsg, wParam, lParam);
}

void WINAPI run_bot(LPVOID param = NULL)
{
	// Since Super Hexagon uses glut32.dll, we can directly use actual glut methods.
	glutSetWindowTitle("You've been gnomed!");
	HWND hwnd = FindWindow(NULL, L"You've been gnomed!");
	
	// Hook the window input handler.
	orig_wnd_proc = (WNDPROC) GetWindowLongPtr(hwnd, GWLP_WNDPROC);
	SetWindowLongPtr(hwnd, GWLP_WNDPROC, (LONG_PTR)&input_handler);

	// Create a console shell for debugging purposes.
	AllocConsole();
	FILE* _f;
	freopen_s(&_f, "CONOUT$", "w", stdout);
	printf("Hello, world\n");
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


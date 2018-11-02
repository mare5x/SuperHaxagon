#include "stdafx.h"
#include "win_console.h"
#include <cstdio>


FILE* p_cout;

void open_console()
{
	AllocConsole();
	freopen_s(&p_cout, "CONOUT$", "w", stdout);
}

void close_console()
{
	// To close the opened console window we must fclose(stdout), however doing that
	// makes future calls to printf (while the console is closed) crash. If we don't 
	// fclose(stdout), the console window stays open until a new console is opened, but
	// we can call printf (without printing anything) ...
	
	// Solution: use freopen to redirect stdout to NUL.

	// "The freopen_s function closes the file currently associated with stream and reassigns stream to the file specified by path."
	// by using NUL, we redirect output to null
	freopen_s(&p_cout, "NUL", "w", stdout);

	FreeConsole();
}

void hide_console()
{
	if (IsWindowVisible(GetConsoleWindow()))
		ShowWindow(GetConsoleWindow(), SW_HIDE);
}

void show_console()
{
	if (!IsWindowVisible(GetConsoleWindow()))
		ShowWindow(GetConsoleWindow(), SW_SHOW);
}

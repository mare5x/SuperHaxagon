// dllmain.cpp : Defines the entry point for the DLL application.
#include "stdafx.h"
#include "SuperHaxagon.h"


void WINAPI run_bot(HMODULE dll)
{
	SuperHaxagon::hook(dll);
}


BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
	if (ul_reason_for_call == DLL_PROCESS_ATTACH) {
		HANDLE thread = CreateThread(NULL, NULL, (LPTHREAD_START_ROUTINE)&run_bot, hModule, NULL, NULL);
		CloseHandle(thread);
	}
    return TRUE;
}


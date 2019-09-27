#include <Windows.h>
#include <cstdio>
#include <cassert>


HMODULE load_dll(HANDLE proc, const char* dll_path)
{
	// write the dll path to process memory 
	size_t path_len = strlen(dll_path) + 1;
	LPVOID remote_string_address = VirtualAllocEx(proc, NULL, path_len, MEM_COMMIT, PAGE_EXECUTE);
	WriteProcessMemory(proc, remote_string_address, dll_path, path_len, NULL);

	// get the address of the LoadLibrary()
	HMODULE k32 = GetModuleHandleA("kernel32.dll");
	LPVOID load_library_adr = GetProcAddress(k32, "LoadLibraryA");

	// create the thread
	HANDLE thread = CreateRemoteThread(proc, NULL, NULL, (LPTHREAD_START_ROUTINE)load_library_adr, remote_string_address, NULL, NULL);

	// finish and clean up
	WaitForSingleObject(thread, INFINITE);

	DWORD dll_handle;
	GetExitCodeThread(thread, &dll_handle);

	CloseHandle(thread);

	VirtualFreeEx(proc, remote_string_address, path_len, MEM_RELEASE);

	return (HMODULE)dll_handle;
}


void unload_dll(HANDLE proc, HMODULE dll_handle)
{
	// get the address of FreeLibrary()
	HMODULE k32 = GetModuleHandleA("kernel32.dll");
	LPVOID free_library_adr = GetProcAddress(k32, "FreeLibrary");

	HANDLE thread = CreateRemoteThread(proc, NULL, NULL, (LPTHREAD_START_ROUTINE)free_library_adr, dll_handle, NULL, NULL);

	WaitForSingleObject(thread, INFINITE);

	DWORD exit_code;
	GetExitCodeThread(thread, &exit_code);

	CloseHandle(thread);
}


int main(int argc, char** argv, char** env)
{
	auto hWnd = FindWindow(nullptr, "Super Hexagon");
	assert(hWnd);

	DWORD processId = -1;
	GetWindowThreadProcessId(hWnd, &processId);
	assert(processId > 0);

	auto const hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, processId);
	assert(hProcess);

	// For LoadLibrary to find the dll path, the dll must either be in the working directory
	// of the process (SuperHexagon.exe's installation folder) or the path must be absolute.
	HMODULE dll_handle = load_dll(hProcess, argc > 1 ? argv[1] : "SuperHaxagon.dll");
	if (dll_handle)
		printf("%x\n", dll_handle);
	else
		printf("Error injecting DLL! Make sure the path is absolute.\n");

	CloseHandle(hProcess);

	return 0;
}

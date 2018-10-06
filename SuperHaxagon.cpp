/*
This is free and unencumbered software released into the public domain by sku.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org/>
*/

#include <Windows.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <type_traits>


#define WRITABLE (PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)


HMODULE load_dll(HANDLE proc, const wchar_t* dll_path)
{
	// write the dll path to process memory 
	size_t path_len = wcslen(dll_path) + 1;
	LPVOID remote_string_address = VirtualAllocEx(proc, NULL, path_len * 2, MEM_COMMIT, PAGE_EXECUTE);
	WriteProcessMemory(proc, remote_string_address, dll_path, path_len * 2, NULL);

	// get the address of the LoadLibrary()
	HMODULE k32 = GetModuleHandleA("kernel32.dll");
	LPVOID load_library_adr = GetProcAddress(k32, "LoadLibraryW");

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
	// get the address of the FreeLibrary()
	HMODULE k32 = GetModuleHandleA("kernel32.dll");
	LPVOID free_library_adr = GetProcAddress(k32, "FreeLibrary");

	HANDLE thread = CreateRemoteThread(proc, NULL, NULL, (LPTHREAD_START_ROUTINE)free_library_adr, dll_handle, NULL, NULL);

	WaitForSingleObject(thread, INFINITE);

	DWORD exit_code;
	GetExitCodeThread(thread, &exit_code);

	CloseHandle(thread);
}


DWORD isArrayMatch(HANDLE const proc, DWORD addr, SIZE_T segment_size, std::vector<BYTE> arr, SIZE_T arr_size)
{
	std::vector<BYTE> proc_arr(segment_size);

	if (ReadProcessMemory(proc, (LPCVOID)addr, proc_arr.data(), segment_size, NULL) == 0) {
		printf("Failed to read memory: %u\n", GetLastError());
		return 0;
	}
	auto it = std::search(std::begin(proc_arr), std::end(proc_arr), std::begin(arr), std::end(arr));
	if (it != std::end(proc_arr)) return addr + std::distance(proc_arr.begin(), it);
	else return 0;
}


DWORD scanSegments(HANDLE const proc, std::vector<BYTE> arr, SIZE_T size)
{
	MEMORY_BASIC_INFORMATION meminfo;
	LPCVOID addr = 0;
	DWORD result = 0; 

	while (1) {
		if (VirtualQueryEx(proc, addr, &meminfo, sizeof(meminfo)) == 0) break;

		if ((meminfo.State & MEM_COMMIT) && (meminfo.Protect & WRITABLE) && (meminfo.Type & MEM_PRIVATE) && !(meminfo.Protect & PAGE_GUARD)) {
			std::cout << meminfo.BaseAddress << std::endl;
			result = isArrayMatch(proc, (DWORD)meminfo.BaseAddress, meminfo.RegionSize, arr, size);
			if (result != 0) return result;
		}
		addr = (unsigned char*)meminfo.BaseAddress + meminfo.RegionSize;
	}
	return 0;
}


struct Memory
{
	HANDLE const hProcess;

	Memory(HANDLE const hProcess)
		: hProcess(hProcess)
	{ }

	~Memory()
	{
		if (hProcess) {
			CloseHandle(hProcess);
		}
	}

	inline DWORD Read(DWORD address) const
	{
		DWORD data = 0;
		ReadProcessMemory(hProcess, reinterpret_cast<LPCVOID>(address), &data, sizeof(DWORD), NULL);

		return data;
	}

	inline DWORD& Read(DWORD address, DWORD& data) const
	{
		SIZE_T numRead = -1;
		auto success = ReadProcessMemory(hProcess, reinterpret_cast<LPCVOID>(address), 
			&data, sizeof(DWORD), &numRead);

		return data;
	}

	void ReadBytes(DWORD address, void* buffer, SIZE_T length) const
	{
		SIZE_T numRead = -1;
		auto success = ReadProcessMemory(hProcess, reinterpret_cast<LPCVOID>(address),
			buffer, length, &numRead);
	}

	template <typename T>
	inline void Write(T address, T data) const
	{
		static_assert(std::is_pod<T>::value, "T must be plain old data.");

		SIZE_T numWritten = -1;
		auto success = WriteProcessMemory(hProcess, reinterpret_cast<LPVOID>(address),
			&data, sizeof(T), &numWritten);
	}
};


struct SuperHexagonApi
{
#pragma pack(push, 1)
	struct Wall
	{
		DWORD slot;
		DWORD distance;  // Should be signed integer as well!
		BYTE enabled;  // Actually, these 4 bytes make up a signed integer containing the width of a wall!
		BYTE fill1[3];  // Actually, these 4 bytes make up a signed integer containing the width of a wall!
		DWORD unk2;
		DWORD unk3;
	};
#pragma pack(pop)

	static_assert(sizeof(Wall) == 20, "Wall struct must be 0x14 bytes total.");

	// ASLR is off in Super Hexagon.
	struct Offsets
	{
		enum : DWORD
		{
			NumSlots = 0x1BC,
			NumWalls = 0x2930,
			FirstWall = 0x220,
			PlayerAngle = 0x2958,
			PlayerAngle2 = 0x2954,
			MouseDownLeft = 0x42858,
			MouseDownRight = 0x4285A,
			MouseDown = 0x42C45,
			WorldAngle = 0x1AC
		};
	};

	DWORD appBase;
	Memory const& memory;
	std::vector<Wall> walls;

	SuperHexagonApi(Memory const& memory)
		: memory(memory)
	{
		// magic numbers array that finds the base pointer
		std::vector<BYTE> arr = { 0x9C, 0xEF, 0x5D, 0, 0x88, 0xEF, 0x5D, 0 };
		appBase = scanSegments(memory.hProcess, arr, arr.size());
		assert(appBase != 0);
	}

	DWORD GetNumSlots() const
	{
		return memory.Read(appBase + Offsets::NumSlots);
	}

	DWORD GetNumWalls() const
	{
		return memory.Read(appBase + Offsets::NumWalls);
	}

	void UpdateWalls()
	{
		walls.clear();
		auto const numWalls = GetNumWalls();
		walls.resize(numWalls);
		memory.ReadBytes(appBase + Offsets::FirstWall, walls.data(), sizeof(Wall) * numWalls);
	}

	DWORD GetPlayerAngle() const
	{
		return memory.Read(appBase + Offsets::PlayerAngle);
	}

	void SetPlayerSlot(DWORD slot) const
	{
		// Move into the center of a given slot number.
		DWORD const angle = 360 / GetNumSlots() * (slot % GetNumSlots()) + (180 / GetNumSlots());
		memory.Write(appBase + Offsets::PlayerAngle, angle);
		memory.Write(appBase + Offsets::PlayerAngle2, angle);
	}

	DWORD GetPlayerSlot() const
	{
		float const angle = static_cast<float>(GetPlayerAngle());
		return static_cast<DWORD>(angle / 360.0f * GetNumSlots());
	}

	void StartMovingLeft() const
	{
		memory.Write<BYTE>(appBase + Offsets::MouseDownLeft, 1);
		memory.Write<BYTE>(appBase + Offsets::MouseDown, 1);
	}

	void StartMovingRight() const
	{
		memory.Write<BYTE>(appBase + Offsets::MouseDownRight, 1);
		memory.Write<BYTE>(appBase + Offsets::MouseDown, 1);
	}

	void ReleaseMouse() const
	{
		memory.Write<BYTE>(appBase + Offsets::MouseDownLeft, 0);
		memory.Write<BYTE>(appBase + Offsets::MouseDownRight, 0);
		memory.Write<BYTE>(appBase + Offsets::MouseDown, 0);
	}

	DWORD GetWorldAngle() const
	{
		return memory.Read(appBase + Offsets::WorldAngle);
	}

	void SetWorldAngle(DWORD angle) const
	{
		memory.Write<DWORD>(appBase + Offsets::WorldAngle, angle);
	}
};

int main(int argc, char** argv, char** env)
{
	auto hWnd = FindWindow(nullptr, "Super Hexagon");
	assert(hWnd);

	DWORD processId = -1;
	GetWindowThreadProcessId(hWnd, &processId);
	assert(processId > 0);

	auto const hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, processId);
	assert(hProcess);

	HMODULE dll_handle = load_dll(hProcess, L"c:\\users\\mare5\\projects\\hacking\\SuperHaxagon\\Debug\\SuperHaxagonDLL.dll");
	printf("%x\n", dll_handle);

	CloseHandle(hProcess);

	return 0;

	Memory const memory(hProcess);
	SuperHexagonApi api(memory);

	for (;;) {
		api.UpdateWalls();
		if (!api.walls.empty()) {
			auto const numSlots = api.GetNumSlots();
			std::vector<DWORD> minDistances(numSlots, -1);

			std::for_each(api.walls.begin(), api.walls.end(), [&] (SuperHexagonApi::Wall const& a) {
				if (a.distance > 0 && a.enabled) {
					minDistances[a.slot % numSlots] = min(minDistances[a.slot % numSlots], a.distance);
				}
			});

			// find slot furthest away (safest)
			auto const maxElement = std::max_element(minDistances.begin(), minDistances.end());
			DWORD const targetSlot = static_cast<DWORD>(std::distance(minDistances.begin(), maxElement));
			std::cout << "Moving to slot [" << targetSlot << "]; world angle is: " << api.GetWorldAngle() << ".\n";

			// TODO: Move properly instead of teleporting around; requires some more wall processing logic.
			api.SetPlayerSlot(targetSlot);
		}

		Sleep(10);
		system("cls"); // Oh the humanity.
	}

	return 0;
}

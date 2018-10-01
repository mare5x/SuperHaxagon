// Common functions related to a process' memory.

#pragma once
#include <Windows.h>
#include <array>
#include <vector>


const size_t JMP_SIZE = 5;  // bytes


struct BasicHookInfo
{
	DWORD hook_at;
	std::vector<BYTE> original_bytes;
};

struct MemoryRegion {
	MemoryRegion() : base_adr(0), size(0) {}
	MemoryRegion(DWORD _base_adr, SIZE_T _size) : base_adr(_base_adr), size(_size) {}

	DWORD base_adr;
	SIZE_T size;

	bool valid() { return size > 0; }

	DWORD end() { return base_adr + size; }
};

template<typename T>
T read_memory(DWORD address)
{
	return *((T*)address);
}

template<typename T>
void read_memory(DWORD address, T* buffer, size_t size)
{
	memcpy(buffer, (T*)address, size);
}


template<typename T>
void write_memory(DWORD address, T value)
{
	*((T*)address) = value;
}


template<typename T>
void write_memory(DWORD address, const T* buffer, size_t size)
{
	memcpy((T*)address, buffer, size);
}


template<typename T>
T* point_memory(DWORD address)
{
	return (T*(address));
}


template<typename T>
DWORD protect_memory(DWORD hook_at, DWORD protection, size_t size)
{
	DWORD old_protection;
	VirtualProtect((LPVOID)hook_at, sizeof(T) * size, protection, &old_protection);
	return old_protection;
}


template<typename T>
DWORD protect_memory(DWORD hook_at, DWORD protection)
{
	return protect_memory<T>(hook_at, protection, 1);
}


/* Writes size bytes of instructions in buffer to address using the PAGE_EXECUTE_READWRITE protection. */
void write_code_buffer(DWORD address, const BYTE* buffer, size_t size);


/* Returns the address of the func_idx-th function in the class vtable pointed to by class_adr. */
DWORD get_VF(DWORD class_adr, DWORD func_idx);


/** Change the address of func_idx in class_adr's vtable with new_func.
Returns the original function address replaced by new_func.
*/
DWORD hook_vtable(DWORD class_adr, DWORD func_idx, DWORD new_func);


/** Place a jump at hook_at to jump_to. size is the number of bytes to be replaced. 
	Returns the address of the instructions after the jump hook.
*/
DWORD jump_hook(DWORD hook_at, DWORD jump_to, size_t size);


/* Unhooks the jump hook placed at hook_at with original_bytes. */
void jump_unhook(DWORD hook_at, const BYTE* original_bytes, size_t size);


/** Place a detour hook at hook_at. 
	Seamlessly redirect/detour execution at hook_at to go to detour. 
	At detour, execute whatever you want, then it executes the length of original
	code replaced by the JMP and jumps back to the original code.
*/
BYTE* detour_hook(DWORD hook_at, DWORD detour, size_t length);


void remove_detour_hook(DWORD hook_at, const BYTE* original, size_t length);


/* Write size bytes of NOPs at hook_at. */
void nop_fill(DWORD hook_at, size_t size);


/* Returns the next memory page after base_adr that has the given permissions (default = PAGE_EXECUTE_READWRITE). */
MemoryRegion next_memory_page(DWORD base_adr, DWORD protection = PAGE_EXECUTE_READWRITE);


MemoryRegion first_memory_page(DWORD protection = PAGE_EXECUTE_READWRITE);


/* Returns the base address of the first matched signature in a memory page with the given protection, or 0 if it doesn't exist. */
DWORD find_signature(const BYTE signature[], size_t size, DWORD protection = PAGE_EXECUTE_READWRITE);
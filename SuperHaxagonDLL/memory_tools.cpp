#include "stdafx.h"
#include "memory_tools.h"

const size_t BUFFER_SIZE = 4096;  // bytes


void write_code_buffer(DWORD address, const BYTE * buffer, size_t size)
{
	DWORD old_protection = protect_memory<BYTE>(address, PAGE_EXECUTE_READWRITE, size);

	write_memory<BYTE>(address, buffer, size);

	protect_memory<BYTE>(address, old_protection, size);
}


DWORD get_VF(DWORD class_adr, DWORD func_idx)
{
	// each (virtual) class has a vtable (table of addresses to functions)
	// a vtable is shared by all instances of the same class
	DWORD vtable = read_memory<DWORD>(class_adr);
	DWORD hook_adr = vtable + func_idx * sizeof(DWORD);
	return read_memory<DWORD>(hook_adr);
}


/** Change the address of func_idx in class_adr's vtable with new_func.
Returns the original function address replaced by new_func.
*/
DWORD hook_vtable(DWORD class_adr, DWORD func_idx, DWORD new_func)
{
	DWORD vtable = read_memory<DWORD>(class_adr);
	DWORD hook_at = vtable + func_idx * sizeof(DWORD);

	DWORD old_protection = protect_memory<DWORD>(hook_at, PAGE_READWRITE);
	DWORD original_func = read_memory<DWORD>(hook_at);
	write_memory<DWORD>(hook_at, new_func);
	protect_memory<DWORD>(hook_at, old_protection);

	return original_func;
}


DWORD jump_hook(DWORD hook_at, DWORD jump_to, size_t size)
{
	DWORD offset = jump_to - hook_at - JMP_SIZE;

	DWORD old_protection = protect_memory<BYTE>(hook_at, PAGE_EXECUTE_READWRITE, size);

	write_memory<BYTE>(hook_at, 0xE9);
	write_memory<DWORD>(hook_at + 1, offset);

	for (size_t i = JMP_SIZE; i < size; ++i)
		write_memory<BYTE>(hook_at + i, 0x90);

	protect_memory<BYTE>(hook_at, old_protection, size);

	return hook_at + JMP_SIZE;
}


void jump_unhook(DWORD hook_at, const BYTE * original_bytes, size_t size)
{
	write_code_buffer(hook_at, original_bytes, size);
}


/* Delete[] the returned buffer. */
BYTE* detour_hook(DWORD hook_at, DWORD detour, size_t length)
{
	BYTE* post_detour_cave = new BYTE[length + JMP_SIZE];	 // a code cave directly in the target process' memory
	memcpy(post_detour_cave, (BYTE*)hook_at, length);  // copy original code
	post_detour_cave[length] = 0xE9;					 // add JMP back to original code (hook_at + ...)
	*(DWORD*)(post_detour_cave + length + 1) = (hook_at + length) - ((DWORD)(post_detour_cave)+length + 5);

	DWORD old_protection = protect_memory<BYTE[JMP_SIZE]>(hook_at, PAGE_EXECUTE_READWRITE);
	write_memory<BYTE>(hook_at, 0xE9);  // JMP hook to detour from hook_at
	write_memory<DWORD>(hook_at + 1, detour - hook_at - 5);
	protect_memory<BYTE[JMP_SIZE]>(hook_at, old_protection);

	DWORD _old_prot;  // make the code cave executable
	VirtualProtect(post_detour_cave, length + 5, PAGE_EXECUTE_READWRITE, &_old_prot);

	return post_detour_cave;
}


void remove_detour_hook(DWORD hook_at, const BYTE* original, size_t length)
{
	DWORD old_protection = protect_memory<BYTE>(hook_at, PAGE_EXECUTE_READWRITE, length);

	write_memory<BYTE>(hook_at, original, length);

	protect_memory<BYTE>(hook_at, old_protection, length);
}

void nop_fill(DWORD hook_at, size_t size)
{
	DWORD old_protection = protect_memory<BYTE>(hook_at, PAGE_EXECUTE_READWRITE, size);

	BYTE* p_memory = (BYTE*)hook_at;
	for (size_t i = 0; i < size; ++i)
		p_memory[i] = 0x90;

	protect_memory<BYTE>(hook_at, old_protection, size);
}

MemoryRegion next_memory_page(DWORD base_adr, DWORD protection)
{
	MEMORY_BASIC_INFORMATION mem_info;
	while (VirtualQuery((LPVOID)base_adr, &mem_info, sizeof(mem_info)) != 0) {
		if ((mem_info.State & MEM_COMMIT) && (mem_info.Protect & protection))
			return MemoryRegion((DWORD)mem_info.BaseAddress, mem_info.RegionSize);
		base_adr += mem_info.RegionSize;
	}
	return MemoryRegion();
}

MemoryRegion first_memory_page(DWORD protection)
{
	return next_memory_page(0, protection);
}

DWORD find_signature(const BYTE signature[], size_t size, DWORD protection)
{
	MemoryRegion page = first_memory_page(protection);
	BYTE buffer[BUFFER_SIZE] = {};

	while (page.valid()) {
		DWORD address = page.base_adr;
		size_t signature_idx = 0;

		do {
			size_t buffer_size = min(BUFFER_SIZE, page.end() - address);
			read_memory<BYTE>(address, buffer, buffer_size);

			for (size_t i = 0; i < buffer_size; ++i) {
				if (signature[signature_idx] == buffer[i]) {
					++signature_idx;
					if (signature_idx == size)
						return address + i + 1 - size;
				}
				else
					signature_idx = 0;
			}

			address += buffer_size;
		} while (address < page.end());

		page = next_memory_page(page.end() + 1, protection);
	}

	return 0;
}
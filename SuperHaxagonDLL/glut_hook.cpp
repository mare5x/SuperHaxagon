#include "stdafx.h"
#include "glut_hook.h"
#include "memory_tools.h"


typedef void(__stdcall* p_glutSwapBuffers)();
p_glutSwapBuffers orig_glutSwapBuffers = nullptr;

const size_t HOOK_SIZE = 9;  // bytes

bool glut_hooked = false;
bool gl_hooked = false;

glut_hook::p_swap_buffers_cb p_swap_buffers_impl = nullptr;

BYTE* post_detour_cave = nullptr;


void __stdcall hooked_swap_buffers()
{
	// This function is executed by Super Hexagon's main thread.
	// That means the thread has an OpenGL context and we can gladLoadGL
	// to get access to all gl functions. (A thread can only make OpenGL 
	// calls if it has a rendering context. - https://docs.microsoft.com/en-us/windows/desktop/opengl/rendering-context-functions,
	// and glad makes gl calls).

	if (!gl_hooked) {
		glut_hook::init_gl();
	}

	if (p_swap_buffers_impl)
		p_swap_buffers_impl();
}

void __declspec(naked) swap_buffers_trampoline()
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

void glut_hook::hook_SwapBuffers(p_swap_buffers_cb func)
{
	p_swap_buffers_impl = func;

	// Hook glutSwapBuffers. To get the function's address we can't just use &glutSwapBuffers, since
	// that version is a stub in this DLL that points to the actual original function in glut32.dll.
	HMODULE glut_handle = GetModuleHandle(L"glut32.dll");
	orig_glutSwapBuffers = (p_glutSwapBuffers)GetProcAddress(glut_handle, "glutSwapBuffers");
	post_detour_cave = detour_hook((DWORD)orig_glutSwapBuffers, (DWORD)&swap_buffers_trampoline, HOOK_SIZE);

	glut_hooked = true;
}

void glut_hook::unhook_SwapBuffers()
{
	remove_detour_hook((DWORD)orig_glutSwapBuffers, post_detour_cave, HOOK_SIZE);
	delete[] post_detour_cave;
	
	glut_hooked = false;
}

bool glut_hook::glut_is_hooked()
{
	return glut_hooked;
}

bool glut_hook::gl_is_hooked()
{
	return gl_hooked;
}

void glut_hook::init_gl()
{
	if (gladLoadGL()) {
		printf("OpenGL: %d.%d\n", GLVersion.major, GLVersion.minor);
        printf("Shader language: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
        glPointSize(4);
		
		gl_hooked = true;
	}
}

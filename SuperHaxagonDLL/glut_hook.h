#pragma once

// If the process uses glut32.dll you can link glut32.lib and use glut functions
// directly in your program. Use hook_SwapBuffers to add custom rendering (or something
// else) code every glutSwapBuffers call.
namespace glut_hook 
{
	typedef void (*p_swap_buffers_cb)();

	void hook_SwapBuffers(p_swap_buffers_cb func);

	void unhook_SwapBuffers();

	bool glut_is_hooked();
	bool gl_is_hooked();

	void init_gl();
}
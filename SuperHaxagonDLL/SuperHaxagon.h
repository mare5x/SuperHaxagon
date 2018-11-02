#pragma once
#include <Windows.h>
#include "SuperStruct.h"

namespace SuperHaxagon {
	void hook(HMODULE dll);
	void WINAPI unhook();

	void draw();
	void update();
}
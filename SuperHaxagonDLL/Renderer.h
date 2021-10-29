#pragma once
#include "Shader.h"
#include "glad\glad.h"

class Renderer {
public:
	Renderer();
	~Renderer();

	void init();

	void render();

	void on_resize(int w, int h) { }

	GLuint vao, vbo;
	Shader shader;
};
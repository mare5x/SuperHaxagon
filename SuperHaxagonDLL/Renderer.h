#pragma once
#include "Shader.h"
#include "glad\glad.h"

class Renderer {
public:
	Renderer();
	~Renderer();

	void init();

	void render();

    void on_resize(int w, int h);

	Shader shader;
private:
	GLuint vao, vbo;
    GLuint texture;
    int w, h;
};
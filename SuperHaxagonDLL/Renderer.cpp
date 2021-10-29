#include "stdafx.h"
#include "Renderer.h"

Renderer::Renderer()
{

}

Renderer::~Renderer()
{
	glDeleteBuffers(1, &vbo);
	glDeleteVertexArrays(1, &vao);
}

void Renderer::init()
{
	float quad[] = {
		0.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 0.0f,
		1.0f, 1.0f
	};

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, (void*)(0));

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	shader = Shader("Shaders/quad.vs", "Shaders/quad.fs");
    printf("Shader ID: %d\n", shader.ID);
}

void Renderer::render()
{
    // Since we are hooking, restore the old shader program,
    // otherwise unexpected results occur.
    int old;
    glGetIntegerv(GL_CURRENT_PROGRAM, &old);

	shader.use();
	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);

    glUseProgram(old);
}
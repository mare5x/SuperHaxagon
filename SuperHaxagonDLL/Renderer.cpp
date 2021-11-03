#include "stdafx.h"
#include "Renderer.h"

Renderer::Renderer()
{

}

Renderer::~Renderer()
{
	glDeleteBuffers(1, &vbo);
	glDeleteVertexArrays(1, &vao);
    glDeleteTextures(1, &texture);
    if (shader.ID != -1) glDeleteProgram(shader.ID);
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

    // Textures:
    glGenTextures(1, &texture);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int old;
    glGetIntegerv(GL_CURRENT_PROGRAM, &old);
    shader.use();
    shader.setInt("tex", 0);
    glUseProgram(old);
}

void Renderer::render()
{
    // Copy current screen to texture.
    BYTE* pixels = new BYTE[w * h * 3];
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    delete[] pixels;

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

void Renderer::on_resize(int w, int h)
{
    this->w = w; 
    this->h = h;
    if (shader.ID != -1) {
        int old;
        glGetIntegerv(GL_CURRENT_PROGRAM, &old);
         
        shader.use();
        shader.setVec2("screen_size", w, h); 

        glUseProgram(old);
    }
}

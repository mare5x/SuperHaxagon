#version 330 core
layout (location = 0) in vec2 quad_pos;

out vec2 v_pos;

void main()
{
    v_pos = quad_pos;

    vec2 pos = 2.0 * quad_pos - 1.0;  // quad_pos;
	gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
}
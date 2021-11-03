#version 330 core

uniform vec2 screen_size;
uniform sampler2D tex;

in vec2 v_pos;

out vec4 frag;

void main()
{
    vec2 p = v_pos;
    p.x *= screen_size.x / screen_size.y;
    
    float r = 0.1;
    vec3 color = vec3(1, 0, 0) * step(length(p - vec2(0.5)) - r, r);

	frag = vec4(color, 1);

    frag = texture(tex, v_pos);
}
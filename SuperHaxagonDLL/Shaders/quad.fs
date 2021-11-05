#version 330 core

uniform vec2 screen_size;
uniform sampler2D tex;

in vec2 v_pos;

out vec4 frag;

vec2 texel_size;

float gray(vec2 uv)
{
    vec4 rgb = texture(tex, uv);
    return (rgb.r + rgb.g + rgb.b) / 3.0;
}

float sobel(vec2 uv)
{
    float dx = 0.0;
    float dy = 0.0;

    dx += gray(uv + vec2(-1.0,  1.0) * texel_size) * -1.0;
    dx += gray(uv + vec2(-1.0,  0.0) * texel_size) * -2.0;
    dx += gray(uv + vec2(-1.0, -1.0) * texel_size) * -1.0;
    dx += gray(uv + vec2(1.0,   1.0) * texel_size) *  1.0;
    dx += gray(uv + vec2(1.0,   0.0) * texel_size) *  2.0;
    dx += gray(uv + vec2(1.0,  -1.0) * texel_size) *  1.0;

    dy += gray(uv + vec2(-1.0,  1.0) * texel_size) *  1.0;
    dy += gray(uv + vec2(0.0,   1.0) * texel_size) *  2.0;
    dy += gray(uv + vec2(1.0,   1.0) * texel_size) *  1.0;
    dy += gray(uv + vec2(-1.0, -1.0) * texel_size) * -1.0;
    dy += gray(uv + vec2(0.0,  -1.0) * texel_size) * -2.0;
    dy += gray(uv + vec2(1.0,  -1.0) * texel_size) * -1.0;

    return sqrt(dx*dx + dy*dy);
}

void main()
{
    texel_size = 1.0 / textureSize(tex, 0);

    // vec2 p = v_pos;
    // p.x *= screen_size.x / screen_size.y;
    
    vec3 color;
    float gradient = sobel(v_pos);
    gradient = sqrt(gradient);  // Boost low values a bit
    color = texture(tex, v_pos).rgb;
    color *= gradient;  // Keep original edge color (could also set it to whatever we want)
    frag = vec4(color, 1);
}
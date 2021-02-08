#version 330 core
layout(location = 0) out vec4 f_out;

uniform sampler2D inTexture;

in vec2 TexCoords;

void main()
{
    f_out = texture(inTexture, TexCoords);
}

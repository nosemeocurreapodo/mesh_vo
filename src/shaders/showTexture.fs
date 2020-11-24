#version 330 core
layout(location = 0) out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D frame;
uniform int lvl;

void main()
{
    //FragColor = texelFetch(frame, ivec2(gl_FragCoord.xy), lvl);
    //FragColor = textureLod(frame, TexCoords, lvl)/255.0 + vec4(0.5,0.5,0.5,1.0);
    FragColor = textureLod(frame, TexCoords, lvl);
}

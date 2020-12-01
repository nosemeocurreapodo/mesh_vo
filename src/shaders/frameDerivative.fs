#version 330 core
layout(location = 0) out vec2 FragColor;

in vec2 v_TexCoord;

// texture samplers
uniform sampler2D frame;

uniform float dx;
uniform float dy;

void main()
{    

    float h_derivative = (texture(frame, v_TexCoord + vec2( dx,0.0)).r - texture(frame, v_TexCoord + vec2(-dx,0.0)).r)/2.0;
    //aca esta al revez, tiene que ver con el sistema de coordenadas de opencv vs en de opengl
    float v_derivative = (texture(frame, v_TexCoord + vec2(0.0,-dy)).r - texture(frame, v_TexCoord + vec2(0.0, dy)).r)/2.0;
    
    FragColor = vec2(1.0,0.0);//vec2(h_derivative, v_derivative);
}

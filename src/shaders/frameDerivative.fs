#version 330 core
layout(location = 0) out vec2 FragColor;

// texture samplers
uniform sampler2D frame;
uniform int lvl;

void main()
{    

    float h_derivative = (texelFetch(frame, ivec2(gl_FragCoord.xy) + ivec2(1.0,0.0), lvl).r - texelFetch(frame, ivec2(gl_FragCoord.xy) + ivec2(-1.0,0.0), lvl).r)/2.0;
    float v_derivative = (texelFetch(frame, ivec2(gl_FragCoord.xy) + ivec2(0.0,1.0), lvl).r - texelFetch(frame, ivec2(gl_FragCoord.xy) + ivec2(0.0,-1.0), lvl).r)/2.0;
    
    FragColor = vec2(h_derivative, v_derivative);
}

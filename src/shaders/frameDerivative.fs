#version 330 core
layout(location = 0) out vec2 FragColor;

// texture samplers
uniform sampler2D frame;

void main()
{    

    float h_derivative = (texture(frame, gl_FragCoord.xy + vec2(1.0,0.0)).r - texture(frame, gl_FragCoord.xy + vec2(-1.0,0.0)).r)/2.0;
    float v_derivative = (texture(frame, gl_FragCoord.xy + vec2(0.0,1.0)).r - texture(frame, gl_FragCoord.xy + vec2(0.0,-1.0)).r)/2.0;
    
    FragColor = vec2(h_derivative, v_derivative);
}

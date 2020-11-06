#version 330 core
layout(location = 0) out vec4 f_color;

noperspective in vec2 g_u_frame;
noperspective in vec2 g_u_keyframe;
in float g_depth;

uniform sampler2D keyframe;
uniform sampler2D frame;
uniform int lvl;

void main()
{
    float f_pixel = textureLod(frame, g_u_frame, lvl).x;
    float kf_pixel = textureLod(keyframe, g_u_keyframe, lvl).x;

    float f_residual = f_pixel*0.5 + kf_pixel*0.5;

    f_color = vec4(f_pixel/255.0,kf_pixel/255.0,0.0,1.0);
}

#version 330 core
layout(location = 0) out float f_residual;

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

    //f_residual = kf_pixel;
    f_residual = f_pixel;
    //f_residual = 1.0/g_depth;
    //f_residual = pow(f_pixel-kf_pixel,2.0);
}

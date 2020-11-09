#version 330 core
layout(location = 0) out float f_residual;

noperspective in vec2 g_u_frame;
in vec2 g_u_keyframe;
in float g_depth;

uniform sampler2D keyframe;
uniform sampler2D frame;
uniform int lvl;

void main()
{
    float f_pixel = textureLod(frame, g_u_frame, lvl).x;
    float kf_pixel = textureLod(keyframe, g_u_keyframe, lvl).x;

    if(f_pixel < 0.0 || kf_pixel < 0.0)
      discard;

    f_residual = pow(f_pixel-kf_pixel,2.0);
}

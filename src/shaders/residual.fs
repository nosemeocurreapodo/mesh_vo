#version 330 core
layout(location = 0) out float f_residual;

noperspective in vec2 v_u_frame;
in vec2 v_u_keyframe;

uniform sampler2D keyframe;
uniform sampler2D frame;
uniform int lvl;

void main()
{
    float f_pixel = textureLod(frame, gl_FragCoord.xy, lvl).x;
    float kf_pixel = textureLod(keyframe, gl_FragCoord.xy, lvl).x;

    //float kf_pixel = textureLod(keyframe, v_u_keyframe, lvl).x;

    f_residual = kf_pixel;//pow(f_pixel-kf_pixel,2.0);
}

#version 330 core
layout(location = 0) out float f_residual;
layout(location = 1) out vec3 f_tra;
layout(location = 2) out vec3 f_rot;

noperspective in vec2 g_uframe;
in vec2 g_ukeyframe;
in vec3 g_pframe;

uniform sampler2D keyframe;
uniform sampler2D frame;
uniform sampler2D frameDer;
uniform int lvl;

uniform mat3 K;

void main()
{
    float f_pixel = textureLod(frame, g_uframe, lvl).x;
    vec2 f_der = textureLod(frameDer, g_uframe, lvl).xy;
    float kf_pixel = textureLod(keyframe, g_ukeyframe, lvl).x;

    if(kf_pixel < 0.0 || f_pixel < 0.0)
      discard;

    float id = 1.0/g_pframe.z;

    float v0 = f_der.x * K[0][0] * id;
    float v1 = f_der.y * K[1][1] * id;
    float v2 = -(v0 * g_pframe.x + v1 * g_pframe.y) * id;

    f_tra = vec3(v0, v1, v2);
    f_rot = vec3( -g_pframe.z * v1 + g_pframe.y * v2, g_pframe.z * v0 - g_pframe.x * v2, -g_pframe.y * v0 + g_pframe.x * v1);

    f_residual = f_pixel - kf_pixel;
}

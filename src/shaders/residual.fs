#version 330 core
layout(location = 0) out float f_residual;

in vec2 v_uframeTexCoord;
in vec2 v_ukeyframeTexCoord;
in vec3 v_pframe;
in vec3 v_pkeyframe;

uniform sampler2D keyframe;
uniform sampler2D frame;

uniform mat3 K;
uniform int lvl;
uniform float width;
uniform float height;

void main()
{
    vec2 uframe = (K*(v_pframe.xyz/v_pframe.z)).xy;
    vec2 ukeyframe = (K*(v_pkeyframe.xyz/v_pkeyframe.z)).xy;
    vec2 uframeTexCoord = vec2(uframe.x/width, 1.0-uframe.y/height);
    vec2 ukeyframeTexCoord = vec2(ukeyframe.x/width, 1.0-ukeyframe.y/height);

    float f_pixel = texture(frame, uframeTexCoord).x;
    float kf_pixel = texture(keyframe, ukeyframeTexCoord).x;

    f_residual = pow(f_pixel-kf_pixel,2.0);
}

#version 330 core
layout(location = 0) out float f_residual;
layout(location = 1) out vec3 f_tra;
layout(location = 2) out vec3 f_rot;

in vec3 v_pframe;
in vec3 v_pkeyframe;

uniform sampler2D keyframe;
uniform sampler2D frame;
uniform sampler2D frameDer;

uniform mat3 K;
uniform float dx;
uniform float dy;

void main()
{
    vec2 uframe = (K*(v_pframe.xyz/v_pframe.z)).xy;
    vec2 ukeyframe = (K*(v_pkeyframe.xyz/v_pkeyframe.z)).xy;

    vec2 uframeTexCoord = vec2(uframe.x*dx, 1.0-uframe.y*dy);
    vec2 ukeyframeTexCoord = vec2(ukeyframe.x*dx, 1.0-ukeyframe.y*dy);

    float f_pixel = texture(frame, uframeTexCoord).x;
    vec2 f_der = texture(frameDer, uframeTexCoord).xy;
    float kf_pixel = texture(keyframe, ukeyframeTexCoord).x;
    vec3 pframe = v_pframe;

    if(kf_pixel < 0.0 || f_pixel < 0.0)
      discard;

    float id = 1.0/pframe.z;

    float v0 = f_der.x * K[0][0] * id;
    float v1 = f_der.y * K[1][1] * id;
    float v2 = -(v0 * pframe.x + v1 * pframe.y) * id;

    f_tra = vec3(v0, v1, v2);
    f_rot = vec3( -pframe.z * v1 + pframe.y * v2, pframe.z * v0 - pframe.x * v2, -pframe.y * v0 + pframe.x * v1);

    f_residual = f_pixel - kf_pixel;
}

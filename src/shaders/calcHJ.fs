#version 330 core
layout(location = 0) out float f_residual;
layout(location = 1) out vec3 f_tra;
layout(location = 2) out vec3 f_rot;

in vec3 v_pkeyframe;

uniform sampler2D keyframe;
uniform sampler2D frame;
uniform sampler2D frameDer;

uniform mat4 framePose;

uniform mat3 invK;
uniform mat3 K;
uniform float dx;
uniform float dy;

void main()
{
    //calculo u y depth con respecto al keyframe
    //vec2 ukf = vec2(gl_FragCoord.x,1.0/dy-gl_FragCoord.y);
    //dkf = gl_FragCoord.z/gl_FragCoord.w;

    vec3 pframe = (framePose * vec4(v_pkeyframe,1.0)).xyz;

    //me esta causando nands en los frame buffers esto, no se porque
    //if(v_pframe.z <= 0.0 || v_pkeyframe.z <= 0.0)
    //  discard;

    vec2 uframe = (K*(pframe.xyz/pframe.z)).xy;
    vec2 ukeyframe = vec2(gl_FragCoord.x,1.0/dy-gl_FragCoord.y);

    vec2 uframeTexCoord = vec2(uframe.x*dx, 1.0-uframe.y*dy);
    vec2 ukeyframeTexCoord = vec2(ukeyframe.x*dx, 1.0-ukeyframe.y*dy);

    float f_pixel = texture(frame, uframeTexCoord).x;
    vec2 f_der = texture(frameDer, uframeTexCoord).xy;
    float kf_pixel = texture(keyframe, ukeyframeTexCoord).x;

    //este no me causa nands, muy extraño!
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

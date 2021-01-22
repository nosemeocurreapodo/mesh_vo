#version 330 core
layout(location = 0) out float f_residual;
layout(location = 1) out vec3 f_tra;
layout(location = 2) out vec3 f_rot;

in vec3 v_pframe;
in vec3 v_pkeyframe;

uniform sampler2D keyframe;
uniform sampler2D keyframeDer;
uniform sampler2D frame;
uniform sampler2D frameDer;

uniform float fx;
uniform float fy;
uniform float cx;
uniform float cy;

uniform float dx;
uniform float dy;

void main()
{
    //calculo u y depth con respecto al keyframe
    //vec2 ukf = vec2(gl_FragCoord.x,1.0/dy-gl_FragCoord.y);
    //dkf = gl_FragCoord.z/gl_FragCoord.w;

    //me esta causando nands en los frame buffers esto, no se porque
    //if(v_pframe.z <= 0.0 || v_pkeyframe.z <= 0.0)
    //  discard;

    vec2 ukeyframe = vec2(fx*v_pkeyframe.x/v_pkeyframe.z+cx,fy*v_pkeyframe.y/v_pkeyframe.z+cy);
    vec2 uframe = vec2(gl_FragCoord.x,1.0/dy-gl_FragCoord.y);

    vec2 ukeyframeTexCoord = vec2(ukeyframe.x*dx, 1.0-ukeyframe.y*dy);
    vec2 uframeTexCoord = vec2(uframe.x*dx, 1.0-uframe.y*dy);

    float f_pixel = texture(frame, uframeTexCoord).x;
    vec2 f_der = texture(frameDer, uframeTexCoord).xy;
    float kf_pixel = texture(keyframe, ukeyframeTexCoord).x;
    vec2 kf_der = texture(keyframeDer, ukeyframeTexCoord).xy;

    //este no me causa nands, muy extraño!
    //if(kf_pixel < 0.0 || f_pixel < 0.0)
    //  discard;

    float id = 1.0/v_pframe.z;

    float v0 = f_der.x * fx * id;
    float v1 = f_der.y * fy * id;
    float v2 = -(v0 * v_pframe.x + v1 * v_pframe.y) * id;

    f_tra = vec3(v0, v1, v2);
    f_rot = vec3( -v_pframe.z * v1 + v_pframe.y * v2, v_pframe.z * v0 - v_pframe.x * v2, -v_pframe.y * v0 + v_pframe.x * v1);

    f_residual = f_pixel - kf_pixel;
}

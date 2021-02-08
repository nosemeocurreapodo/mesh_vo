#version 330 core
layout(location = 0) out vec4 f_j1;
layout(location = 1) out vec4 f_j2;
layout(location = 2) out vec4 f_j3;
layout(location = 3) out vec4 f_j4;
layout(location = 4) out vec4 f_j5;
layout(location = 5) out vec4 f_j6;
layout(location = 6) out vec4 f_j7;

in vec3 v_pframe;
in vec3 v_pkeyframe;

uniform sampler2D keyframe;
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
    //vec2 kf_der = texture(keyframeDer, ukeyframeTexCoord).xy;

    //este no me causa nands, muy extraño!
    //if(kf_pixel < 0.0 || f_pixel < 0.0)
    //  discard;

    float id = 1.0/v_pframe.z;

    float v0 = f_der.x * fx * id;
    float v1 = f_der.y * fy * id;
    float v2 = -(v0 * v_pframe.x + v1 * v_pframe.y) * id;

    vec3 tra = vec3(v0, v1, v2);
    vec3 rot = vec3( -v_pframe.z * v1 + v_pframe.y * v2, v_pframe.z * v0 - v_pframe.x * v2, -v_pframe.y * v0 + v_pframe.x * v1);

    float residual = f_pixel - kf_pixel;

    f_j1 = vec4(tra.x,tra.y,tra.z,rot.x)*residual;
    f_j2 = vec4(rot.y*residual,rot.z*residual,tra.x*tra.x,tra.x*tra.y);
    f_j3 = vec4(tra.x*tra.z,tra.x*rot.x,tra.x*rot.y,tra.x*rot.z);
    f_j4 = vec4(tra.y*tra.y,tra.y*tra.z,tra.y*rot.x,tra.y*rot.y);
    f_j5 = vec4(tra.y*rot.z,tra.z*tra.z,tra.z*rot.x,tra.z*rot.y);
    f_j6 = vec4(tra.z*rot.z,rot.x*rot.x,rot.x*rot.y,rot.x*rot.z);
    f_j7 = vec4(rot.y*rot.y,rot.y*rot.z,rot.z*rot.z,0.0);
}

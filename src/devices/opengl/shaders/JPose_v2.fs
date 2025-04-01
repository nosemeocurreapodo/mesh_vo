#version 330 core
layout(location = 0) out vec4 gradient1;
layout(location = 1) out vec4 gradient2;
layout(location = 2) out vec4 hessian1;
layout(location = 3) out vec4 hessian2;
layout(location = 4) out vec4 hessian3;
layout(location = 5) out vec4 hessian4;
layout(location = 6) out vec4 hessian5;
layout(location = 7) out vec4 hessian6;

in vec2 v_u;
in vec3 v_pkeyframe;
in vec3 v_pframe;

uniform sampler2D keyframe;
uniform sampler2D frame;
uniform sampler2D frameDer;

uniform mat4 framePose;

uniform float fx;
uniform float fy;
uniform float cx;
uniform float cy;

uniform float dx;
uniform float dy;

void main()
{
    //if(v_pframe.z < 0.0)
    //  discard;

    //if(v_u.x < 0.0 || v_u.x > 1.0/dx || v_u.y < 0.0 || v_u.y > 1.0/dy)
    //  discard;

    //from frame perspective
    vec2 ukeyframe = vec2(v_u.x,1.0/dy-v_u.y);
    vec2 uframe  = vec2(gl_FragCoord.x,gl_FragCoord.y);
    vec3 pframe = (framePose*vec4(v_pkeyframe,1.0)).xyz;

    //from keyframe perspective
    //vec2 uframe = vec2(v_u.x,1.0/dy-v_u.y);
    //vec2 ukeyframe  = vec2(gl_FragCoord.x,gl_FragCoord.y);

    vec2 ukeyframeTexCoord = vec2(ukeyframe.x*dx, ukeyframe.y*dy);
    vec2 uframeTexCoord = vec2(uframe.x*dx, uframe.y*dy);

    float f_pixel = texture(frame, uframeTexCoord).x*255.0;
    vec2 f_der = texture(frameDer, uframeTexCoord).xy;
    float kf_pixel = texture(keyframe, ukeyframeTexCoord).x*255.0;
    //vec2 kf_der = texture(keyframeDer, ukeyframeTexCoord).xy;

    float id = 1.0/pframe.z;

    float v0 = f_der.x * fx * id;
    float v1 = f_der.y * fy * id;
    float v2 = -(v0 * pframe.x + v1 * pframe.y) * id;

    vec3 tra = vec3(v0, v1, v2);
    vec3 rot = vec3( -pframe.z * v1 + pframe.y * v2, pframe.z * v0 - pframe.x * v2, -pframe.y * v0 + pframe.x * v1);

    float residual = f_pixel - kf_pixel;

    float J[6];
    J[0] = tra.x;
    J[1] = tra.y;
    J[2] = tra.z;
    J[3] = rot.x;
    J[4] = rot.y;
    J[5] = rot.z;

    gradient1.x = J[0]*residual;
    gradient1.y = J[1]*residual;
    gradient1.z = J[2]*residual;
    gradient2.x = J[3]*residual;
    gradient2.y = J[4]*residual;
    gradient2.z = J[5]*residual;

    hessian1.x = J[0]*J[0];
    hessian1.y = J[0]*J[1];
    hessian1.z = J[0]*J[2];
    hessian1.w = J[0]*J[3];
    hessian2.x = J[0]*J[4];
    hessian2.y = J[0]*J[5];

    hessian2.z = J[1]*J[1];
    hessian2.w = J[1]*J[2];
    hessian3.x = J[1]*J[3];
    hessian3.y = J[1]*J[4];
    hessian3.z = J[1]*J[5];

    hessian3.w = J[2]*J[2];
    hessian4.x = J[2]*J[3];
    hessian4.y = J[2]*J[4];
    hessian4.z = J[2]*J[5];

    hessian4.w = J[3]*J[3];
    hessian5.x = J[3]*J[4];
    hessian5.y = J[3]*J[5];

    hessian5.z = J[4]*J[4];
    hessian5.w = J[4]*J[5];

    hessian6.x = J[5]*J[5];
}

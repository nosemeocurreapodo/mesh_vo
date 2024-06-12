#version 330 core
layout(location = 0) out float f_residual;
layout(location = 1) out vec4 f_tra;
layout(location = 2) out vec4 f_rot;

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

    f_tra = vec4(v0, v1, v2, 0.0);
    f_rot = vec4( -pframe.z * v1 + pframe.y * v2, pframe.z * v0 - pframe.x * v2, -pframe.y * v0 + pframe.x * v1, 0.0);

    f_residual = f_pixel - kf_pixel;
}

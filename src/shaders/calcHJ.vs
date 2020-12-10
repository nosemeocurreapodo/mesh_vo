#version 330 core
layout (location = 0) in vec3 p;

uniform mat4 framePose;
uniform mat4 projection;
uniform mat4 opencv2opengl;

out vec3 v_pkeyframe;

uniform float fxinv;
uniform float fyinv;
uniform float cxinv;
uniform float cyinv;

void main()
{
    vec2 ukeyframe = vec2(p.x,p.y);
    float depthKeyframe = p.z;

    vec4 pkeyframe = vec4(vec3(fxinv*ukeyframe.x + cxinv, fyinv*ukeyframe.y + cyinv, 1.0)*depthKeyframe,1.0);
    vec4 pframe = framePose * pkeyframe;

    gl_Position = projection * opencv2opengl * pframe;

    v_pkeyframe = pkeyframe.xyz;
}

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
    vec3 pkeyframe = vec3(p.x,p.y,1.0)*p.z;

    vec4 pframe = framePose * vec4(pkeyframe,1.0);

    gl_Position = projection * opencv2opengl * pframe;

    v_pkeyframe = pkeyframe.xyz;
}

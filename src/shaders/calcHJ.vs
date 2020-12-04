#version 330 core
layout (location = 0) in vec3 p;

uniform mat4 keyframePose;
uniform mat4 framePose;
uniform mat4 projection;
uniform mat4 opencv2opengl;

out vec3 v_pkeyframe;

void main()
{
    vec4 pworld = vec4(p, 1.0);
    vec4 pframe = framePose * pworld;
    vec4 pkeyframe = keyframePose * pworld;

    gl_Position = projection * opencv2opengl * pframe;

    v_pkeyframe = pkeyframe.xyz;
}

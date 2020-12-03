#version 330 core
layout (location = 0) in vec3 p;

out vec3 v_pkeyframe;

uniform mat4 keyframePose;
uniform mat4 projection;
uniform mat4 opencv2opengl;

void main()
{
    vec4 pworld = vec4(p, 1.0);
    vec4 pkeyframe = keyframePose * pworld;

    gl_Position = projection * opencv2opengl * pkeyframe;

    v_pkeyframe = pkeyframe.xyz;
}

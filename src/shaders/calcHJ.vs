#version 330 core
layout (location = 0) in vec3 p;

out vec3 v_pframe;
out vec3 v_pkeyframe;

uniform mat4 framePose;
uniform mat4 keyframePose;
uniform mat4 projection;
uniform mat4 opencv2opengl;

void main()
{
    //vec2 u = vec2(p.x*width, p.y*height);
    //float depth = 1.0/p.z;
    //vec4 pworld = vec4(invK*(vec3(u,1.0)*depth), 1.0f);

    vec4 pworld = vec4(p, 1.0);
    vec4 pframe = framePose * pworld;
    vec4 pkeyframe = keyframePose * pworld;

    gl_Position = projection * opencv2opengl * pframe;

    v_pframe = pframe.xyz;
    v_pkeyframe = pkeyframe.xyz;
}

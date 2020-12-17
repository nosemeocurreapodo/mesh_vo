#version 330 core
layout (location = 0) in vec3 p;

out vec3 v_pframe;
out vec2 v_uframe;
out vec3 v_pkeyframe;

flat out int v_vertexID;

uniform mat4 framePose;
uniform mat4 projection;
uniform mat4 opencv2opengl;

uniform float fx;
uniform float fy;
uniform float cx;
uniform float cy;

uniform float fxinv;
uniform float fyinv;
uniform float cxinv;
uniform float cyinv;

void main()
{
    vec3 rkeyframe = vec3(p.x,p.y,1.0);
    float dkeyframe = exp(p.z);

    vec3 pkeyframe = rkeyframe*dkeyframe;

    vec4 pframe = framePose * vec4(pkeyframe,1.0);

    vec2 uframe = vec2(fx*pframe.x/pframe.z+cx,fy*pframe.y/pframe.z+cy);

    gl_Position = projection * opencv2opengl * pframe;

    v_pframe = pframe.xyz;
    v_uframe = uframe;

    v_pkeyframe = pkeyframe.xyz;

    v_vertexID = gl_VertexID;
}

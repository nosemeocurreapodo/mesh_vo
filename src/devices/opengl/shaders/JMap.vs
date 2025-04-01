#version 330 core
layout (location = 0) in vec3 p;

out vec3 v_pkeyframe;
out vec3 v_pframe;
out vec2 v_u;
out vec2 v_u2;
flat out float v_idepth;
flat out int v_vertexID;

uniform mat4 framePose;
uniform mat4 projection;
uniform mat4 opencv2opengl;

uniform float dx;
uniform float dy;

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
    vec3 pkeyframe = vec3(p.x,p.y,1.0)/p.z;
    vec4 pframe = framePose*vec4(pkeyframe,1.0);

    v_pframe = pframe.xyz;
    v_pkeyframe = pkeyframe.xyz;

    v_idepth = p.z;
    v_vertexID = gl_VertexID;

    //from frame perspective
    v_u = vec2(fx*p.x+cx, fy*p.y+cy);
    v_u2 = vec2(fx*pframe.x/pframe.z+cx, fy*pframe.y/pframe.z+cy);
    gl_Position = projection * opencv2opengl * pframe;

    //from keyframe perspective
    //v_u = vec2(fx*pframe.x/pframe.z+cx, fy*pframe.y/pframe.z+cy);
    //gl_Position = projection * opencv2opengl * vec4(pkeyframe,1.0);
}

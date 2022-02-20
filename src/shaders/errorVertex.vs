#version 330 core
layout (location = 0) in vec3 p;

out vec2 v_u;
out vec2 v_u2;
flat out int v_vertexID;

uniform mat4 framePose;
uniform mat4 projection;
uniform mat4 opencv2opengl;

uniform float fx;
uniform float fy;
uniform float cx;
uniform float cy;

uniform float dx;
uniform float dy;

void main()
{
    vec3 pkeyframe = vec3(p.x,p.y,1.0)/p.z;
    vec4 pframe = framePose * vec4(pkeyframe,1.0);

    //from frame perspective
    gl_Position = projection * opencv2opengl * pframe;
    v_u = vec2(fx*p.x+cx, fy*p.y+cy);
    v_u2 = vec2(fx*pframe.x/pframe.z+cx, fy*pframe.y/pframe.z+cy);
    v_vertexID = gl_VertexID;

    //from keyframe perspective
    //gl_Position = projection * opencv2opengl * vec4(pkeyframe,1.0);
    //v_u = vec2(fx*pframe.x/pframe.z+cx, fy*pframe.y/pframe.z+cy);
}

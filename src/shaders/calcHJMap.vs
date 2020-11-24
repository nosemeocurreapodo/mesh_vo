#version 330 core
layout (location = 0) in vec3 p;

out vec2 v_ukf; //no lineal en If
out vec3 v_uf; //lineal
out float v_dkf; // no lineal
out vec3 v_pw; //no lineal
out vec3 v_pc; //no lineal

flat out int v_vertexID; //flat

uniform mat3 K;
uniform mat3 invK;
uniform mat4 cameraPose;
uniform mat4 projection;
uniform mat4 opencv2opengl;
uniform float width;
uniform float height;

void main()
{
    //vec2 ukf = vec2(p.x*width,p.y*height);
    //float dkf = 1.0/p.z;
    //vec4 pworld = vec4(invK*(vec3(ukf,1.0)*dkf), 1.0f);

    vec4 pworld = vec4(p, 1.0);
    vec4 pframe = framePose * pworld;
    vec4 pkeyframe = keyframePose * pworld;

    gl_Position = projection * opencv2opengl * pframe;

    vec2 uframe = (K*(pframe.xyz/pframe.z)).xy;
    vec2 ukeyframe = (K*(pkeyframe.xyz/pkeyframe.z)).xy;

    v_ukeyframe = ukeyframe;
    v_uframe = uframe;
    v_dkf = dkf;
    v_pw = pworld.xyz;
    v_pc = pcamera.xyz;

    v_vertexID = gl_VertexID;
}

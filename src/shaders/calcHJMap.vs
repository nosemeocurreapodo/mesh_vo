#version 330 core
layout (location = 0) in vec3 p;

out vec2 v_ukf; //no lineal en If
out vec3 v_uf; //lineal
out float v_dkf; // no lineal
out vec3 v_pw; //no lineal
out vec3 v_pc; //no lineal
flat out mat3 v_KRK_1; //flat
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
    vec2 ukf = vec2(p.x*width,p.y*height);
    float dkf = 1.0/p.z;
    vec4 pworld = vec4(invK*(vec3(ukf,1.0)*dkf), 1.0f);
    vec4 pcamera = cameraPose*pworld;
    gl_Position = projection * opencv2opengl * pcamera; // esto seria p (up) para opengl

    vec2 uf = K*((pcamera).xyz);

    mat3 rotation = mat3(cameraPose);
    vec3 KRK_1 = K*rotation*invK;

    v_ukf = ukf;
    v_uf = uf;
    v_dkf = dkf;
    v_pw = pworld.xyz;
    v_pc = pcamera.xyz;
    v_KRK_1 = KRK_1;
    v_vertexID = gl_VertexID;
}

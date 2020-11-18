#version 330 core
layout (location = 0) in vec3 p;

out vec2 v_u; //no lineal en If
out vec2 v_uTexCoord;
out float v_depth; // no lineal
out vec3 v_pw; //no lineal
out vec3 v_pk; //lineal
out vec3 v_d_pk_d_z; //lineal
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
    vec2 u = p.xy;
    float depth = 1.0/p.z;
    vec4 pworld = vec4(invK*(vec3(u,1.0)*depth), 1.0f);
    vec4 pcamera = cameraPose*pworld;
    vec3 pk = K*((pcamera).xyz);
    gl_Position = projection * opencv2opengl * pcamera; // esto seria p (up) para opengl

    //calculo d_pk_d_z
    mat3 rotation = mat3(cameraPose);
    vec3 d_pk_d_z = K*rotation*invK*vec3(u,1.0f);

    vec2 uTexCoord = vec2(u.x/width, 1.0-u.y/height);
    v_u = u;
    v_uTexCoord = uTexCoord;
    v_depth = depth;
    v_pk = pk;
    v_pw = pworld.xyz;
    v_d_pk_d_z = d_pk_d_z;
    v_vertexID = gl_VertexID;
}

#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec2 v_ukf[];
in vec3 v_uf[];
in float v_dkf[];
in vec3 v_pw[];
in vec3 v_pc[];
flat in mat3 v_KRK_1[];
flat in int v_vertexID[];

out vec2 g_ukf;
out vec3 g_uf;
out float g_dkf;
out vec3 g_pw; // depende de depth, depth no es lineal
out vec3 g_pc;

flat out vec3 g_KRK_1;

flat out vec3 g_normal;
flat out int g_vertexID[3];

flat out vec3 g_d_N_d_p0[3];
flat out vec3 g_d_N_d_p1[3];
flat out vec3 g_d_N_d_p2[3];
flat out vec3 g_p1[3];

uniform mat3 invK;

void main() {

    gl_PrimitiveID = gl_PrimitiveIDIn;
    vec3 normal = cross(v_pw[1] - v_pw[0], v_pw[2] - v_pw[0]);

    g_normal = normal;///length(normal);
    g_KRK_1 = v_KRK_1[0];

    vec3 pw2mpw1;
    //p = u, v, idepth
    pw2mpw1 = (v_pw[2]-v_pw[1]);
    g_d_N_d_p0[0] = vec3(0.0,pw2mpw1.z*invK[0][0]*v_dkf[0],-invK[0][0]*pw2mpw1.y*v_dkf[0]);
    g_d_N_d_p0[1] = vec3(-invK[1][1]*pw2mpw1.z*v_dkf[0],0.0,pw2mpw1.x*invK[1][1]*v_dkf[0]);
    g_d_N_d_p0[2] = cross(pw2mpw1, -invK*vec3(v_ukf[0],1.0)*(v_dkf[0]*v_dkf[0]));
    g_p1[0] = v_pw[1];

    pw2mpw1 = (v_pw[0]-v_pw[2]);
    g_d_N_d_p1[0] = vec3(0.0,pw2mpw1.z*invK[0][0]*v_depth[1],-invK[0][0]*pw2mpw1.y*v_depth[1]);
    g_d_N_d_p1[1] = vec3(-invK[1][1]*pw2mpw1.z*v_depth[1],0.0,pw2mpw1.x*invK[1][1]*v_depth[1]);
    g_d_N_d_p1[2] = cross(pw2mpw1, -invK*vec3(v_ukf[1],1.0)*(v_depth[1]*v_depth[1]));
    g_p1[1] = v_pw[2];

    pw2mpw1 = (v_pw[1]-v_pw[0]);
    g_d_N_d_p2[0] = vec3(0.0,pw2mpw1.z*invK[0][0]*v_depth[2],-invK[0][0]*pw2mpw1.y*v_depth[2]);
    g_d_N_d_p2[1] = vec3(-invK[1][1]*pw2mpw1.z*v_depth[2],0.0,pw2mpw1.x*invK[1][1]*v_depth[2]);
    g_d_N_d_p2[2] = cross(pw2mpw1, -invK*vec3(v_ukf[2],1.0)*(v_depth[2]*v_depth[2]));
    g_p1[2] = v_pw[0];

    // vertice 0
    g_vertexID[0] = v_vertexID[0];
    g_ukf = v_ukf[0];
    g_uf = v_uf[0];
    g_depth = v_depth[0];
    g_pw = v_pw[0];

    gl_Position = gl_in[0].gl_Position;
    EmitVertex();

    //vertice 1
    g_vertexID[1] = v_vertexID[1];
    g_ukf = v_ukf[1];
    g_uf = v_uf[1];
    g_depth = v_depth[1];
    g_pw = v_pw[1];

    gl_Position = gl_in[1].gl_Position;
    EmitVertex();

    //vertice 2
    g_vertexID[2] = v_vertexID[2];
    g_ukf = v_ukf[2];
    g_uf = v_uf[2];
    g_depth = v_depth[2];
    g_pw = v_pw[2];

    gl_Position = gl_in[2].gl_Position;
    EmitVertex();

    EndPrimitive();
}

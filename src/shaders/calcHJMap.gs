#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec3 v_pframe[];
in vec3 v_pkeyframe[];
in vec3 v_pworld[];

flat in int v_vertexID[];

out vec2 g_pframe;
out vec3 g_pkeyframe;
out float g_pworld;

flat out int g_vertexID[3];

flat out vec3 g_N_p0;
flat out mat3 g_d_N_d_p0;
flat out vec3 g_pr_p0;
flat out vec3 g_N_p1;
flat out mat3 g_d_N_d_p1;
flat out vec3 g_pr_p1;
flat out vec3 g_N_p2;
flat out mat3 g_d_N_d_p2;
flat out vec3 g_pr_p2;

void main() {

    gl_PrimitiveID = gl_PrimitiveIDIn;
    vec3 pw2mpw1 = vec3(0.0);

    g_N_p0 = cross(v_pworld[0] - v_pworld[1], v_pworld[2] - v_pworld[1]);
    g_pr_p0 = v_pworld[1];

    pw2mpw1 = (v_pworld[2]-v_pworld[1]);
    g_d_N_d_p0[0] = vec3(0.0,-pw2mpw1.z,pw2mpw1.y);
    g_d_N_d_p0[1] = vec3(pw2mpw1.z,0.0,-pw2mpw1.x);
    g_d_N_d_p0[2] = vec3(-pw2mpw1.y,pw2mpw1.x,0.0);

    g_N_p1 = cross(v_pworld[1] - v_pworld[0], v_pworld[2] - v_pworld[0]);
    g_pr_p1 = v_pworld[0];

    pw2mpw1 = (v_pworld[2]-v_pworld[0]);
    g_d_N_d_p1[0] = vec3(0.0,-pw2mpw1.z,pw2mpw1.y);
    g_d_N_d_p1[1] = vec3(pw2mpw1.z,0.0,-pw2mpw1.x);
    g_d_N_d_p1[2] = vec3(-pw2mpw1.y,pw2mpw1.x,0.0);

    g_N_p2 = cross(v_pworld[2] - v_pworld[1], v_pworld[0] - v_pworld[1]);
    g_pr_p2 = v_pworld[1];

    pw2mpw1 = (v_pworld[0]-v_pworld[1]);
    g_d_N_d_p2[0] = vec3(0.0,-pw2mpw1.z,pw2mpw1.y);
    g_d_N_d_p2[1] = vec3(pw2mpw1.z,0.0,-pw2mpw1.x);
    g_d_N_d_p2[2] = vec3(-pw2mpw1.y,pw2mpw1.x,0.0);

    // vertice 0
    g_vertexID[0] = v_vertexID[0];
    g_pkeyframe = v_pkeyframe[0];
    g_pframe = v_pframe[0];
    g_pworld = v_pworld[0];

    gl_Position = gl_in[0].gl_Position;
    EmitVertex();

    //vertice 1
    g_vertexID[1] = v_vertexID[1];
    g_pkeyframe = v_pkeyframe[1];
    g_pframe = v_pframe[1];
    g_pworld = v_pworld[1];

    gl_Position = gl_in[1].gl_Position;
    EmitVertex();

    //vertice 2
    g_vertexID[2] = v_vertexID[2];
    g_pkeyframe = v_pkeyframe[2];
    g_pframe = v_pframe[2];
    g_pworld = v_pworld[2];

    gl_Position = gl_in[2].gl_Position;
    EmitVertex();

    EndPrimitive();
}

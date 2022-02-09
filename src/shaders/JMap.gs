#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec3 v_pkeyframe[];
in vec3 v_pframe[];
in vec2 v_u[];
flat in float v_idepth[];
flat in int v_vertexID[];

out vec3 g_pframe;
out vec3 g_pkeyframe;
out vec2 g_u;
flat out float g_idepth[3];
flat out int g_vertexID[3];

flat out vec3 g_N_p0;
flat out vec3 g_d_N_d_z0;
flat out float g_N_p0_dot_point;
flat out vec3 g_pr_p0;

flat out vec3 g_N_p1;
flat out vec3 g_d_N_d_z1;
flat out float g_N_p1_dot_point;
flat out vec3 g_pr_p1;

flat out vec3 g_N_p2;
flat out vec3 g_d_N_d_z2;
flat out float g_N_p2_dot_point;
flat out vec3 g_pr_p2;

uniform float dx;
uniform float dy;

void main() {

    if(v_pframe[0].z <= 0.0 || v_pframe[1].z <= 0.0 || v_pframe[2].z <= 0.0)
      return;

    gl_PrimitiveID = gl_PrimitiveIDIn;
    g_vertexID[0] = v_vertexID[0];
    g_vertexID[1] = v_vertexID[1];
    g_vertexID[2] = v_vertexID[2];

    g_idepth[0] = v_idepth[0];
    g_idepth[1] = v_idepth[1];
    g_idepth[2] = v_idepth[2];

    vec3 pw2mpw1 = vec3(0.0);

    g_N_p0 = cross(v_pkeyframe[0] - v_pkeyframe[1], v_pkeyframe[2] - v_pkeyframe[1]);
    pw2mpw1 = (v_pkeyframe[2]-v_pkeyframe[1]);
    g_d_N_d_z0 =  cross(v_pkeyframe[0], pw2mpw1);
    g_N_p0_dot_point = dot(g_N_p0,v_pkeyframe[1]);
    g_pr_p0 = v_pkeyframe[1];

    g_N_p1 = cross(v_pkeyframe[1] - v_pkeyframe[0], v_pkeyframe[2] - v_pkeyframe[0]);
    pw2mpw1 = (v_pkeyframe[2]-v_pkeyframe[0]);
    g_d_N_d_z1 = cross(v_pkeyframe[1], pw2mpw1);
    g_N_p1_dot_point = dot(g_N_p1,v_pkeyframe[0]);
    g_pr_p1 = v_pkeyframe[0];

    g_N_p2 = cross(v_pkeyframe[2] - v_pkeyframe[1], v_pkeyframe[0] - v_pkeyframe[1]);
    pw2mpw1 = (v_pkeyframe[0]-v_pkeyframe[1]);
    g_d_N_d_z2 = cross(v_pkeyframe[2], pw2mpw1);
    g_N_p2_dot_point = dot(g_N_p2,v_pkeyframe[1]);
    g_pr_p2 = v_pkeyframe[1];

    //if(abs(g_N_p0_dot_point) <= 0.0 || abs(g_N_p1_dot_point) <= 0.0 || abs(g_N_p2_dot_point) <= 0.0)
    //   return;

    // vertice 0

    g_pkeyframe = v_pkeyframe[0];
    g_pframe = v_pframe[0];
    g_u = v_u[0];

    gl_Position = gl_in[0].gl_Position;
    EmitVertex();

    //vertice 1
    g_pkeyframe = v_pkeyframe[1];
    g_pframe = v_pframe[1];
    g_u = v_u[1];

    gl_Position = gl_in[1].gl_Position;
    EmitVertex();

    //vertice 2
    g_pkeyframe = v_pkeyframe[2];
    g_pframe = v_pframe[2];
    g_u = v_u[2];

    gl_Position = gl_in[2].gl_Position;
    EmitVertex();

    EndPrimitive();
}

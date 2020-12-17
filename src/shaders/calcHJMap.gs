#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec3 v_pframe[];
in vec2 v_uframe[];
in vec3 v_pkeyframe[];

flat in int v_vertexID[];

out vec3 g_pframe;
out vec3 g_pkeyframe;

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

    if(v_uframe[0].x < 0.0 || v_uframe[0].x > 1/dx || v_uframe[0].y < 0.0 || v_uframe[0].y > 1/dy)
        return;
    if(v_uframe[1].x < 0.0 || v_uframe[1].x > 1/dx || v_uframe[1].y < 0.0 || v_uframe[1].y > 1/dy)
        return;
    if(v_uframe[2].x < 0.0 || v_uframe[2].x > 1/dx || v_uframe[2].y < 0.0 || v_uframe[2].y > 1/dy)
        return;

    gl_PrimitiveID = gl_PrimitiveIDIn;
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

    // vertice 0
    g_vertexID[0] = v_vertexID[0];
    g_pkeyframe = v_pkeyframe[0];
    g_pframe = v_pframe[0];

    gl_Position = gl_in[0].gl_Position;
    EmitVertex();

    //vertice 1
    g_vertexID[1] = v_vertexID[1];
    g_pkeyframe = v_pkeyframe[1];
    g_pframe = v_pframe[1];

    gl_Position = gl_in[1].gl_Position;
    EmitVertex();

    //vertice 2
    g_vertexID[2] = v_vertexID[2];
    g_pkeyframe = v_pkeyframe[2];
    g_pframe = v_pframe[2];

    gl_Position = gl_in[2].gl_Position;
    EmitVertex();

    EndPrimitive();
}

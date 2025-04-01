#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec3 v_pkeyframe[];
in vec2 v_u2[];
flat in int v_vertexID[];

out vec3 g_pkeyframe;
flat out int g_vertexID[3];

uniform float dx;
uniform float dy;

void main() {

    //if(v_pframe[0].z <= 0.0 || v_pframe[1].z <= 0.0 || v_pframe[2].z <= 0.0)
    //  return;
    //if(gl_in[0].gl_Position.x < -1.0 || gl_in[0].gl_Position.x > 1.0 || gl_in[0].gl_Position.y < -1.0 || gl_in[0].gl_Position.y > 1.0)
    //    return;

    if(v_u2[0].x < 0.0 || v_u2[0].x > 1.0/dx || v_u2[0].y < 0.0 || v_u2[0].y > 1.0/dy)
        return;
    if(v_u2[1].x < 0.0 || v_u2[1].x > 1.0/dx || v_u2[1].y < 0.0 || v_u2[1].y > 1.0/dy)
        return;
    if(v_u2[2].x < 0.0 || v_u2[2].x > 1.0/dx || v_u2[2].y < 0.0 || v_u2[2].y > 1.0/dy)
        return;

    g_vertexID[0] = v_vertexID[0];
    g_vertexID[1] = v_vertexID[1];
    g_vertexID[2] = v_vertexID[2];

    // vertice 0
    g_pkeyframe = v_pkeyframe[0];
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();

    //vertice 1
    g_pkeyframe = v_pkeyframe[1];
    gl_Position = gl_in[1].gl_Position;
    EmitVertex();

    //vertice 2
    g_pkeyframe = v_pkeyframe[2];
    gl_Position = gl_in[2].gl_Position;
    EmitVertex();

    EndPrimitive();
}

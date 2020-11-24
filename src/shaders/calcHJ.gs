#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

noperspective in vec2 v_uframe[];
in vec2 v_ukeyframe[];
in vec3 v_pframe[];

noperspective out vec2 g_uframe;
out vec2 g_ukeyframe;
out vec3 g_pframe;

void main() {

    // vertice 0
    g_uframe = v_uframe[0];
    g_ukeyframe = v_ukeyframe[0];
    g_pframe = v_pframe[0];

    gl_Position = gl_in[0].gl_Position;
    EmitVertex();

    //vertice 1
    g_uframe = v_uframe[1];
    g_ukeyframe = v_ukeyframe[1];
    g_pframe = v_pframe[1];

    gl_Position = gl_in[1].gl_Position;
    EmitVertex();

    //vertice 2
    g_uframe = v_uframe[2];
    g_ukeyframe = v_ukeyframe[2];
    g_pframe = v_pframe[2];

    gl_Position = gl_in[2].gl_Position;
    EmitVertex();

    EndPrimitive();
}

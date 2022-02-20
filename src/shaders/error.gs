#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec2 v_u[];
in float v_iz[];

out vec2 g_u;
flat out vec3 g_iz_error;

uniform float dx;
uniform float dy;

void main() {

    g_iz_error =  vec3((v_iz[0]-v_iz[1]),(v_iz[0]-v_iz[2]),(v_iz[1]-v_iz[2]));

    // vertice 0
    g_u = v_u[0];
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();

    //vertice 1
    g_u = v_u[1];
    gl_Position = gl_in[1].gl_Position;
    EmitVertex();

    //vertice 2
    g_u = v_u[2];
    gl_Position = gl_in[2].gl_Position;
    EmitVertex();

    EndPrimitive();
}

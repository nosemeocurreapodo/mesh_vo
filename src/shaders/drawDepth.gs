#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec3 v_pw[];
in float v_depth[];

out vec3 g_pw;
out float g_depth;

flat out vec3 g_normal;
flat out int isInitialized;

void main() {

    gl_PrimitiveID = gl_PrimitiveIDIn;
    vec3 normal = cross(v_pw[1] - v_pw[0], v_pw[2] - v_pw[0]);

    if(v_depth[0] > 0.0 && v_depth[1] > 0.0 && v_depth[2] > 0.0 )
        isInitialized = 1;
    else
        isInitialized = 0;

    g_normal = normal;
    // vertice 0
    g_depth = v_depth[0];
    g_pw = v_pw[0];

    gl_Position = gl_in[0].gl_Position;
    EmitVertex();

    //vertice 1
    g_depth = v_depth[1];
    g_pw = v_pw[1];

    gl_Position = gl_in[1].gl_Position;
    EmitVertex();

    //vertice 2
    g_depth = v_depth[2];
    g_pw = v_pw[2];

    gl_Position = gl_in[2].gl_Position;
    EmitVertex();

    EndPrimitive();
}

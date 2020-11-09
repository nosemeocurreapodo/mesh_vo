#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

noperspective in vec2 v_u_frame[];
in vec2 v_u_keyframe[];
in vec3 v_pcamera[];
in float v_depth[];

noperspective out vec2 g_u_frame;
out vec2 g_u_keyframe;
out vec3 g_pcamera;
out float g_depth;

void main() {

    gl_PrimitiveID = gl_PrimitiveIDIn;

    // vertice 0
    g_u_frame = v_u_frame[0];
    g_u_keyframe = v_u_keyframe[0];
    g_pcamera = v_pcamera[0];
    g_depth = v_depth[0];

    gl_Position = gl_in[0].gl_Position;
    EmitVertex();

    //vertice 1
    g_u_frame = v_u_frame[1];
    g_u_keyframe = v_u_keyframe[1];
    g_pcamera = v_pcamera[1];
    g_depth = v_depth[1];

    gl_Position = gl_in[1].gl_Position;
    EmitVertex();

    //vertice 2
    g_u_frame = v_u_frame[2];
    g_u_keyframe = v_u_keyframe[2];
    g_pcamera = v_pcamera[2];
    g_depth = v_depth[2];

    gl_Position = gl_in[2].gl_Position;
    EmitVertex();

    EndPrimitive();
}

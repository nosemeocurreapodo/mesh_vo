#version 330 core
layout (location = 0) in vec3 p;

out float v_depth;

uniform mat3 invK;
uniform mat4 cameraPose;
uniform mat4 projection;
uniform mat4 opencv2opengl;

uniform float width;
uniform float height;

void main()
{
    //vec2 u = vec2(p.x*width, p.y*height);
    //float depth = 1.0/p.z;
    //vec4 pworld = vec4(invK*(vec3(u,1.0)*depth), 1.0f);

    vec4 pworld = vec4(p, 1.0);
    vec4 pcamera = cameraPose * pworld;
    gl_Position = projection * opencv2opengl * pcamera;

    v_depth = pcamera.z;
}

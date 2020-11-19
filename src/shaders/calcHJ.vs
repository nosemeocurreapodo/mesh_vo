#version 330 core
layout (location = 0) in vec3 p;

noperspective out vec2 v_u_frame;
out vec2 v_u_keyframe;
out vec3 v_pcamera;
out float v_depth;


uniform mat3 K;
uniform mat3 invK;
uniform mat4 cameraPose;
uniform mat4 keyframePose;
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
    vec2 ucamera = (K*(pcamera.xyz/pcamera.z)).xy;

    gl_Position = projection * opencv2opengl * pcamera;

    v_u_frame = vec2(ucamera.x/width, 1.0-ucamera.y/height);
    v_u_keyframe = vec2(p.x, 1.0-p.y);

    v_pcamera = pcamera.xyz;
    v_depth = depth;
}

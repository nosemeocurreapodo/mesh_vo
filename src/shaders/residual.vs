#version 330 core
layout (location = 0) in vec3 p;

out vec2 v_uframeTexCoord;
out vec2 v_ukeyframeTexCoord;
out vec3 v_pframe;
out vec3 v_pkeyframe;

uniform mat3 K;
uniform mat3 invK;
uniform mat4 framePose;
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
    vec4 pframe = framePose * pworld;
    vec4 pkeyframe = keyframePose * pworld;

    gl_Position = projection * opencv2opengl * pframe;

    vec2 uframe = (K*(pframe.xyz/pframe.z)).xy;
    vec2 ukeyframe = (K*(pkeyframe.xyz/pkeyframe.z)).xy;

    v_uframeTexCoord = vec2(uframe.x/width, 1.0-uframe.y/height);
    v_ukeyframeTexCoord = vec2(ukeyframe.x/width, 1.0-ukeyframe.y/height);
    v_pframe = pframe.xyz;
    v_pkeyframe = pkeyframe.xyz;
}

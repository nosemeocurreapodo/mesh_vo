#version 330 core
layout(location = 0) out ivec4 f_vertexID;
layout(location = 1) out int f_primitiveID;
layout(location = 2) out float f_error;
layout(location = 3) out vec4 f_d_I_d_p0;
layout(location = 4) out vec4 f_d_I_d_p1;
layout(location = 5) out vec4 f_d_I_d_p2;

in vec2 g_pframe;
in vec3 g_pkeyframe;
in float g_pworld;

flat in int g_vertexID[3];

flat in vec3 g_N_p0;
flat in mat3 g_d_N_d_p0;
flat in vec3 g_pr_p0;
flat in vec3 g_N_p1;
flat in mat3 g_d_N_d_p1;
flat in vec3 g_pr_p1;
flat in vec3 g_N_p2;
flat in mat3 g_d_N_d_p2;
flat in vec3 g_pr_p2;

uniform float width;
uniform float height;

uniform mat3 K;
uniform mat3 invK;

// texture samplers
uniform sampler2D keyframe;
uniform sampler2D keyframeDer
uniform sampler2D frame;
uniform sampler2D frameDer;

void main()
{
    vec2 uframe = (K*(g_pframe.xyz/g_pframe.z)).xy;
    vec2 ukeyframe = (K*(g_pkeyframe.xyz/g_pkeyframe.z)).xy;

    vec2 uframeTexCoord = vec2(uframe.x/width, 1.0-uframe.y/height);
    vec2 ukeyframeTexCoord = vec2(ukeyframe.x/width, 1.0-ukeyframe.y/height);

    float ikeyframe = texture(keyframe,ukeyframeTexCoord).r;
    vec2 dkeyframe = texture(keyframeDer,ukeyframeTexCoord).r;

    float iframe = texture(frame,uframeTexCoord).r;
    vec2 dframe = texture(frameDer,uframeTexCoord).r;

    vec3 d_I_d_pframe = vec3(0);
    d_I_d_pframe.x = dframe.x*K[0][0]/pframe.z;
    d_I_d_pframe.y = dframe.y*K[1][1]/pframe.z;
    d_I_d_pframe.z = -(d_I_d_pframe.x*pframe.x/pframe.z + d_I_d_pframe.y*pframe.y/pframe.z);

    vec3 d_I_d_pkeyframe = vec3(0);
    d_I_d_pkeyframe.x = dkeyframe.x*K[0][0]/pkeyframe.z;
    d_I_d_pkeyframe.y = dkeyframe.y*K[1][1]/pkeyframe.z;
    d_I_d_pkeyframe.z = -(d_I_d_pkeyframe.x*pkeyframe.x/pkeyframe.z + d_I_d_pkeyframe.y*pkeyframe.y/pkeyframe.z);

    vec3 d_pworld_p0x = g_pr_p0 + g_d_N_d_p0.x


    f_vertexID = ivec4(g_vertexID[0], g_vertexID[1], g_vertexID[2], 1);
    f_primitiveID = gl_PrimitiveID;
    f_error = error;
    f_d_I_d_p0 = vec4(d_I_d_p0, 1.0);
    f_d_I_d_p1 = vec4(d_I_d_p1, 1.0);
    f_d_I_d_p2 = vec4(d_I_d_p2, 1.0);
}

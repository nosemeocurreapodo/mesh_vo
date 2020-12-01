#version 330 core
layout(location = 0) out ivec4 f_vertexID;
layout(location = 1) out int f_primitiveID;
layout(location = 2) out float f_error;
layout(location = 3) out vec4 f_d_I_d_p0;
layout(location = 4) out vec4 f_d_I_d_p1;
layout(location = 5) out vec4 f_d_I_d_p2;

in vec3 g_pframe;
in vec3 g_pkeyframe;
in vec3 g_pworld;

flat in int g_vertexID[3];

flat in vec3 g_N_p0;
flat in vec3 g_d_N_d_p0_x;
flat in vec3 g_d_N_d_p0_y;
flat in vec3 g_d_N_d_p0_z;
flat in vec3 g_pr_p0;
flat in vec3 g_N_p1;
flat in vec3 g_d_N_d_p1_x;
flat in vec3 g_d_N_d_p1_y;
flat in vec3 g_d_N_d_p1_z;
flat in vec3 g_pr_p1;
flat in vec3 g_N_p2;
flat in vec3 g_d_N_d_p2_x;
flat in vec3 g_d_N_d_p2_y;
flat in vec3 g_d_N_d_p2_z;
flat in vec3 g_pr_p2;

uniform float dx;
uniform float dy;

uniform mat3 K;
uniform mat3 invK;

uniform mat4 framePose;
uniform mat4 keyframePose;

// texture samplers
uniform sampler2D keyframe;
uniform sampler2D keyframeDer;
uniform sampler2D frame;
uniform sampler2D frameDer;

void main()
{

    //if point not visible
    if(g_pkeyframe.z <= 0.0 || g_pframe.z <= 0.0)
      discard;

    //tambien tendria problemas si la normal tiene algun componente igual a cero, PENSAR ESTE PROBLEMA!
    if(g_N_p0.x == 0.0 || g_N_p0.y == 0.0 || g_N_p0.z == 0.0)
      discard;
    if(g_N_p1.x == 0.0 || g_N_p1.y == 0.0 || g_N_p1.z == 0.0)
      discard;
    if(g_N_p2.x == 0.0 || g_N_p2.y == 0.0 || g_N_p2.z == 0.0)
      discard;

    vec2 uframe = (K*(g_pframe.xyz/g_pframe.z)).xy;
    vec2 ukeyframe = (K*(g_pkeyframe.xyz/g_pkeyframe.z)).xy;

    vec2 uframeTexCoord = vec2(uframe.x*dx, 1.0-uframe.y*dy);
    vec2 ukeyframeTexCoord = vec2(ukeyframe.x*dx, 1.0-ukeyframe.y*dy);

    float ikeyframe = texture(keyframe,ukeyframeTexCoord).x;
    vec2 dkeyframe = texture(keyframeDer,ukeyframeTexCoord).xy;

    float iframe = texture(frame,uframeTexCoord).x;
    vec2 dframe = texture(frameDer,uframeTexCoord).xy;

    if(ikeyframe < 0.0 || iframe < 0.0)
      discard;

    vec3 d_I_d_pframe = vec3(0);
    d_I_d_pframe.x = dframe.x*K[0][0]/g_pframe.z;
    d_I_d_pframe.y = dframe.y*K[1][1]/g_pframe.z;
    d_I_d_pframe.z = -(d_I_d_pframe.x*g_pframe.x/g_pframe.z + d_I_d_pframe.y*g_pframe.y/g_pframe.z);

    vec3 d_I_d_pkeyframe = vec3(0);
    d_I_d_pkeyframe.x = dkeyframe.x*K[0][0]/g_pkeyframe.z;
    d_I_d_pkeyframe.y = dkeyframe.y*K[1][1]/g_pkeyframe.z;
    d_I_d_pkeyframe.z = -(d_I_d_pkeyframe.x*g_pkeyframe.x/g_pkeyframe.z + d_I_d_pkeyframe.y*g_pkeyframe.y/g_pkeyframe.z);

    vec3 prmpw_p0 = (g_pr_p0 - g_pworld);

    //vec3 pworld_p0;
    //pworld_p0.x = g_pr_p0.x + (g_N_p0.y*pr0mpw.y + g_N_p0.z*pr0mpw.z)/g_N_p0.x;
    //pworld_p0.y = g_pr_p0.y + (g_N_p0.x*pr0mpw.x + g_N_p0.z*pr0mpw.z)/g_N_p0.y;
    //pworld_p0.z = g_pr_p0.z + (g_N_p0.x*pr0mpw.x + g_N_p0.y*pr0mpw.y)/g_N_p0.z;

    vec3 d_pworld_p0_x;
    d_pworld_p0_x.x = (g_d_N_d_p0_x.y*prmpw_p0.y + g_d_N_d_p0_x.z*prmpw_p0.z)/g_N_p0.x - g_d_N_d_p0_x.x*(g_N_p0.y*prmpw_p0.y + g_N_p0.z*prmpw_p0.z)/(g_N_p0.x*g_N_p0.x);
    d_pworld_p0_x.y = (g_d_N_d_p0_x.x*prmpw_p0.x + g_d_N_d_p0_x.z*prmpw_p0.z)/g_N_p0.y - g_d_N_d_p0_x.y*(g_N_p0.x*prmpw_p0.x + g_N_p0.z*prmpw_p0.z)/(g_N_p0.y*g_N_p0.y);
    d_pworld_p0_x.z = (g_d_N_d_p0_x.x*prmpw_p0.x + g_d_N_d_p0_x.y*prmpw_p0.y)/g_N_p0.z - g_d_N_d_p0_x.z*(g_N_p0.x*prmpw_p0.x + g_N_p0.y*prmpw_p0.y)/(g_N_p0.z*g_N_p0.z);

    vec3 d_pworld_p0_y;
    d_pworld_p0_y.x = (g_d_N_d_p0_y.y*prmpw_p0.y + g_d_N_d_p0_y.z*prmpw_p0.z)/g_N_p0.x - g_d_N_d_p0_y.x*(g_N_p0.y*prmpw_p0.y + g_N_p0.z*prmpw_p0.z)/(g_N_p0.x*g_N_p0.x);
    d_pworld_p0_y.y = (g_d_N_d_p0_y.x*prmpw_p0.x + g_d_N_d_p0_y.z*prmpw_p0.z)/g_N_p0.y - g_d_N_d_p0_y.y*(g_N_p0.x*prmpw_p0.x + g_N_p0.z*prmpw_p0.z)/(g_N_p0.y*g_N_p0.y);
    d_pworld_p0_y.z = (g_d_N_d_p0_y.x*prmpw_p0.x + g_d_N_d_p0_y.y*prmpw_p0.y)/g_N_p0.z - g_d_N_d_p0_y.z*(g_N_p0.x*prmpw_p0.x + g_N_p0.y*prmpw_p0.y)/(g_N_p0.z*g_N_p0.z);

    vec3 d_pworld_p0_z;
    d_pworld_p0_z.x = (g_d_N_d_p0_z.y*prmpw_p0.y + g_d_N_d_p0_z.z*prmpw_p0.z)/g_N_p0.x - g_d_N_d_p0_z.x*(g_N_p0.y*prmpw_p0.y + g_N_p0.z*prmpw_p0.z)/(g_N_p0.x*g_N_p0.x);
    d_pworld_p0_z.y = (g_d_N_d_p0_z.x*prmpw_p0.x + g_d_N_d_p0_z.z*prmpw_p0.z)/g_N_p0.y - g_d_N_d_p0_z.y*(g_N_p0.x*prmpw_p0.x + g_N_p0.z*prmpw_p0.z)/(g_N_p0.y*g_N_p0.y);
    d_pworld_p0_z.z = (g_d_N_d_p0_z.x*prmpw_p0.x + g_d_N_d_p0_z.y*prmpw_p0.y)/g_N_p0.z - g_d_N_d_p0_z.z*(g_N_p0.x*prmpw_p0.x + g_N_p0.y*prmpw_p0.y)/(g_N_p0.z*g_N_p0.z);


    vec3 prmpw_p1 = (g_pr_p1 - g_pworld);

    vec3 d_pworld_p1_x;
    d_pworld_p1_x.x = (g_d_N_d_p1_x.y*prmpw_p1.y + g_d_N_d_p1_x.z*prmpw_p1.z)/g_N_p1.x - g_d_N_d_p1_x.x*(g_N_p1.y*prmpw_p1.y + g_N_p1.z*prmpw_p1.z)/(g_N_p1.x*g_N_p1.x);
    d_pworld_p1_x.y = (g_d_N_d_p1_x.x*prmpw_p1.x + g_d_N_d_p1_x.z*prmpw_p1.z)/g_N_p1.y - g_d_N_d_p1_x.y*(g_N_p1.x*prmpw_p1.x + g_N_p1.z*prmpw_p1.z)/(g_N_p1.y*g_N_p1.y);
    d_pworld_p1_x.z = (g_d_N_d_p1_x.x*prmpw_p1.x + g_d_N_d_p1_x.y*prmpw_p1.y)/g_N_p1.z - g_d_N_d_p1_x.z*(g_N_p1.x*prmpw_p1.x + g_N_p1.y*prmpw_p1.y)/(g_N_p1.z*g_N_p1.z);

    vec3 d_pworld_p1_y;
    d_pworld_p1_y.x = (g_d_N_d_p1_y.y*prmpw_p1.y + g_d_N_d_p1_y.z*prmpw_p1.z)/g_N_p1.x - g_d_N_d_p1_y.x*(g_N_p1.y*prmpw_p1.y + g_N_p1.z*prmpw_p1.z)/(g_N_p1.x*g_N_p1.x);
    d_pworld_p1_y.y = (g_d_N_d_p1_y.x*prmpw_p1.x + g_d_N_d_p1_y.z*prmpw_p1.z)/g_N_p1.y - g_d_N_d_p1_y.y*(g_N_p1.x*prmpw_p1.x + g_N_p1.z*prmpw_p1.z)/(g_N_p1.y*g_N_p1.y);
    d_pworld_p1_y.z = (g_d_N_d_p1_y.x*prmpw_p1.x + g_d_N_d_p1_y.y*prmpw_p1.y)/g_N_p1.z - g_d_N_d_p1_y.z*(g_N_p1.x*prmpw_p1.x + g_N_p1.y*prmpw_p1.y)/(g_N_p1.z*g_N_p1.z);

    vec3 d_pworld_p1_z;
    d_pworld_p1_z.x = (g_d_N_d_p1_z.y*prmpw_p1.y + g_d_N_d_p1_z.z*prmpw_p1.z)/g_N_p1.x - g_d_N_d_p1_z.x*(g_N_p1.y*prmpw_p1.y + g_N_p1.z*prmpw_p1.z)/(g_N_p1.x*g_N_p1.x);
    d_pworld_p1_z.y = (g_d_N_d_p1_z.x*prmpw_p1.x + g_d_N_d_p1_z.z*prmpw_p1.z)/g_N_p1.y - g_d_N_d_p1_z.y*(g_N_p1.x*prmpw_p1.x + g_N_p1.z*prmpw_p1.z)/(g_N_p1.y*g_N_p1.y);
    d_pworld_p1_z.z = (g_d_N_d_p1_z.x*prmpw_p1.x + g_d_N_d_p1_z.y*prmpw_p1.y)/g_N_p1.z - g_d_N_d_p1_z.z*(g_N_p1.x*prmpw_p1.x + g_N_p1.y*prmpw_p1.y)/(g_N_p1.z*g_N_p1.z);


    vec3 prmpw_p2 = (g_pr_p2 - g_pworld);

    vec3 d_pworld_p2_x;
    d_pworld_p2_x.x = (g_d_N_d_p2_x.y*prmpw_p2.y + g_d_N_d_p2_x.z*prmpw_p2.z)/g_N_p2.x - g_d_N_d_p2_x.x*(g_N_p2.y*prmpw_p2.y + g_N_p2.z*prmpw_p2.z)/(g_N_p2.x*g_N_p2.x);
    d_pworld_p2_x.y = (g_d_N_d_p2_x.x*prmpw_p2.x + g_d_N_d_p2_x.z*prmpw_p2.z)/g_N_p2.y - g_d_N_d_p2_x.y*(g_N_p2.x*prmpw_p2.x + g_N_p2.z*prmpw_p2.z)/(g_N_p2.y*g_N_p2.y);
    d_pworld_p2_x.z = (g_d_N_d_p2_x.x*prmpw_p2.x + g_d_N_d_p2_x.y*prmpw_p2.y)/g_N_p2.z - g_d_N_d_p2_x.z*(g_N_p2.x*prmpw_p2.x + g_N_p2.y*prmpw_p2.y)/(g_N_p2.z*g_N_p2.z);

    vec3 d_pworld_p2_y;
    d_pworld_p2_y.x = (g_d_N_d_p2_y.y*prmpw_p2.y + g_d_N_d_p2_y.z*prmpw_p2.z)/g_N_p2.x - g_d_N_d_p2_y.x*(g_N_p2.y*prmpw_p2.y + g_N_p2.z*prmpw_p2.z)/(g_N_p2.x*g_N_p2.x);
    d_pworld_p2_y.y = (g_d_N_d_p2_y.x*prmpw_p2.x + g_d_N_d_p2_y.z*prmpw_p2.z)/g_N_p2.y - g_d_N_d_p2_y.y*(g_N_p2.x*prmpw_p2.x + g_N_p2.z*prmpw_p2.z)/(g_N_p2.y*g_N_p2.y);
    d_pworld_p2_y.z = (g_d_N_d_p2_y.x*prmpw_p2.x + g_d_N_d_p2_y.y*prmpw_p2.y)/g_N_p2.z - g_d_N_d_p2_y.z*(g_N_p2.x*prmpw_p2.x + g_N_p2.y*prmpw_p2.y)/(g_N_p2.z*g_N_p2.z);

    vec3 d_pworld_p2_z;
    d_pworld_p2_z.x = (g_d_N_d_p2_z.y*prmpw_p2.y + g_d_N_d_p2_z.z*prmpw_p2.z)/g_N_p2.x - g_d_N_d_p2_z.x*(g_N_p2.y*prmpw_p2.y + g_N_p2.z*prmpw_p2.z)/(g_N_p2.x*g_N_p2.x);
    d_pworld_p2_z.y = (g_d_N_d_p2_z.x*prmpw_p2.x + g_d_N_d_p2_z.z*prmpw_p2.z)/g_N_p2.y - g_d_N_d_p2_z.y*(g_N_p2.x*prmpw_p2.x + g_N_p2.z*prmpw_p2.z)/(g_N_p2.y*g_N_p2.y);
    d_pworld_p2_z.z = (g_d_N_d_p2_z.x*prmpw_p2.x + g_d_N_d_p2_z.y*prmpw_p2.y)/g_N_p2.z - g_d_N_d_p2_z.z*(g_N_p2.x*prmpw_p2.x + g_N_p2.y*prmpw_p2.y)/(g_N_p2.z*g_N_p2.z);

    vec3 d_I_d_p0;
    d_I_d_p0.x = dot(d_I_d_pframe,mat3(framePose)*d_pworld_p0_x) - dot(d_I_d_pkeyframe,mat3(keyframePose)*d_pworld_p0_x);
    d_I_d_p0.y = dot(d_I_d_pframe,mat3(framePose)*d_pworld_p0_y) - dot(d_I_d_pkeyframe,mat3(keyframePose)*d_pworld_p0_y);
    d_I_d_p0.z = dot(d_I_d_pframe,mat3(framePose)*d_pworld_p0_z) - dot(d_I_d_pkeyframe,mat3(keyframePose)*d_pworld_p0_z);

    vec3 d_I_d_p1;
    d_I_d_p1.x = dot(d_I_d_pframe,mat3(framePose)*d_pworld_p1_x) - dot(d_I_d_pkeyframe,mat3(keyframePose)*d_pworld_p1_x);
    d_I_d_p1.y = dot(d_I_d_pframe,mat3(framePose)*d_pworld_p1_y) - dot(d_I_d_pkeyframe,mat3(keyframePose)*d_pworld_p1_y);
    d_I_d_p1.z = dot(d_I_d_pframe,mat3(framePose)*d_pworld_p1_z) - dot(d_I_d_pkeyframe,mat3(keyframePose)*d_pworld_p1_z);

    vec3 d_I_d_p2;
    d_I_d_p2.x = dot(d_I_d_pframe,mat3(framePose)*d_pworld_p2_x) - dot(d_I_d_pkeyframe,mat3(keyframePose)*d_pworld_p2_x);
    d_I_d_p2.y = dot(d_I_d_pframe,mat3(framePose)*d_pworld_p2_y) - dot(d_I_d_pkeyframe,mat3(keyframePose)*d_pworld_p2_y);
    d_I_d_p2.z = dot(d_I_d_pframe,mat3(framePose)*d_pworld_p2_z) - dot(d_I_d_pkeyframe,mat3(keyframePose)*d_pworld_p2_z);

    float error = iframe - ikeyframe;

    f_vertexID = ivec4(g_vertexID[0], g_vertexID[1], g_vertexID[2], 1);
    f_primitiveID = gl_PrimitiveID;
    f_error = error;
    //f_d_I_d_p0 = vec4(d_I_d_p0, 1.0);
    f_d_I_d_p0 = vec4(dkeyframe, 0.0, 1.0);
    f_d_I_d_p1 = vec4(d_I_d_p1, 1.0);
    f_d_I_d_p2 = vec4(d_I_d_p2, 1.0);
}

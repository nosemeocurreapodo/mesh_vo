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
flat in vec3 g_Nb_p0;
flat in mat3 g_R_d_N_d_p0;
flat in float g_Nb_p0_dot_point;
flat in vec3 g_pr_p0;

flat in vec3 g_N_p1;
flat in vec3 g_Nb_p1;
flat in mat3 g_R_d_N_d_p1;
flat in float g_Nb_p1_dot_point;
flat in vec3 g_pr_p1;

flat in vec3 g_N_p2;
flat in vec3 g_Nb_p2;
flat in mat3 g_R_d_N_d_p2;
flat in float g_Nb_p2_dot_point;
flat in vec3 g_pr_p2;


uniform float dx;
uniform float dy;

uniform float fx;
uniform float fy;
uniform float cx;
uniform float cy;

uniform float fxinv;
uniform float fyinv;
uniform float cxinv;
uniform float cyinv;

uniform mat4 framePose;
uniform mat4 keyframePose;
uniform mat4 frame2keyframePose;

// texture samplers
uniform sampler2D keyframe;
uniform sampler2D keyframeDer;
uniform sampler2D frame;
uniform sampler2D frameDer;

void main()
{
    //if point not visible
    //if(g_pkeyframe.z <= 0.0 || g_pframe.z <= 0.0)
    //  discard;

    vec2 ukeyframe = vec2(fx*g_pkeyframe.x/g_pkeyframe.z+cx,fy*g_pkeyframe.y/g_pkeyframe.z+cy);
    vec2 uframe = vec2(gl_FragCoord.x,1.0/dy-gl_FragCoord.y);

    vec2 ukeyframeTexCoord = vec2(ukeyframe.x*dx, 1.0-ukeyframe.y*dy);
    vec2 uframeTexCoord = vec2(uframe.x*dx, 1.0-uframe.y*dy);

    float ikeyframe = texture(keyframe,ukeyframeTexCoord).x;
    vec2 dkeyframe = texture(keyframeDer,ukeyframeTexCoord).xy;

    float iframe = texture(frame,uframeTexCoord).x;
    vec2 dframe = texture(frameDer,uframeTexCoord).xy;

    //if(ikeyframe < 0.0 || iframe < 0.0)
    //  discard;

    vec3 d_I_d_pkeyframe = vec3(0);
    d_I_d_pkeyframe.x = dkeyframe.x*fx/g_pkeyframe.z;
    d_I_d_pkeyframe.y = dkeyframe.y*fy/g_pkeyframe.z;
    d_I_d_pkeyframe.z = -(d_I_d_pkeyframe.x*g_pkeyframe.x/g_pkeyframe.z + d_I_d_pkeyframe.y*g_pkeyframe.y/g_pkeyframe.z);

    vec3 rayframe = g_pframe/g_pframe.z;

    vec3 d_pkeyframe_d_z = mat3(frame2keyframePose)*rayframe;

    float d_I_d_z = dot(d_I_d_pkeyframe,d_pkeyframe_d_z);

    float Nb_p0_dot_ray = dot(g_Nb_p0,rayframe);
    float Nb_p1_dot_ray = dot(g_Nb_p1,rayframe);
    float Nb_p2_dot_ray = dot(g_Nb_p2,rayframe);

    vec3 d_z_d_p0 = g_R_d_N_d_p0*(g_pr_p0/Nb_p0_dot_ray - g_Nb_p0_dot_point*rayframe/(Nb_p0_dot_ray*Nb_p0_dot_ray));
    vec3 d_z_d_p1 = g_R_d_N_d_p1*(g_pr_p1/Nb_p1_dot_ray - g_Nb_p1_dot_point*rayframe/(Nb_p1_dot_ray*Nb_p1_dot_ray));
    vec3 d_z_d_p2 = g_R_d_N_d_p2*(g_pr_p2/Nb_p2_dot_ray - g_Nb_p2_dot_point*rayframe/(Nb_p2_dot_ray*Nb_p2_dot_ray));

    vec3 d_I_d_p0 = d_I_d_z*d_z_d_p0;
    vec3 d_I_d_p1 = d_I_d_z*d_z_d_p1;
    vec3 d_I_d_p2 = d_I_d_z*d_z_d_p2;

    float error = ikeyframe - iframe;

    f_vertexID = ivec4(g_vertexID[0], g_vertexID[1], g_vertexID[2], 1);
    f_primitiveID = gl_PrimitiveID;
    f_error = error;
    f_d_I_d_p0 = vec4(d_I_d_p0, 1.0);
    f_d_I_d_p1 = vec4(d_I_d_p1, 1.0);
    f_d_I_d_p2 = vec4(d_I_d_p2, 1.0);
}

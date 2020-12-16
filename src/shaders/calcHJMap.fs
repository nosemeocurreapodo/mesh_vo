#version 330 core
layout(location = 0) out vec3 f_vertexID;
layout(location = 1) out float f_primitiveID;
layout(location = 2) out float f_error;
layout(location = 3) out float f_d_I_d_z0;
layout(location = 4) out float f_d_I_d_z1;
layout(location = 5) out float f_d_I_d_z2;

in vec3 g_pframe;
in vec3 g_pkeyframe;

flat in int g_vertexID[3];

flat in vec3 g_N_p0;
flat in vec3 g_d_N_d_z0;
flat in float g_N_p0_dot_point;
flat in vec3 g_pr_p0;

flat in vec3 g_N_p1;
flat in vec3 g_d_N_d_z1;
flat in float g_N_p1_dot_point;
flat in vec3 g_pr_p1;

flat in vec3 g_N_p2;
flat in vec3 g_d_N_d_z2;
flat in float g_N_p2_dot_point;
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

    //if(abs(dframe.x) < 0.2 || abs(dframe.y) < 0.2)
    //    discard;

    vec3 d_I_d_pframe = vec3(0);
    d_I_d_pframe.x = dframe.x*fx/g_pframe.z;
    d_I_d_pframe.y = dframe.y*fy/g_pframe.z;
    d_I_d_pframe.z = -(d_I_d_pframe.x*g_pframe.x/g_pframe.z + d_I_d_pframe.y*g_pframe.y/g_pframe.z);

    vec3 raykeyframe = g_pkeyframe/g_pkeyframe.z;

    vec3 d_pframe_d_z = mat3(framePose)*raykeyframe;

    float d_I_d_z = dot(d_I_d_pframe,d_pframe_d_z);

    float N_p0_dot_ray = dot(g_N_p0,raykeyframe);
    float N_p1_dot_ray = dot(g_N_p1,raykeyframe);
    float N_p2_dot_ray = dot(g_N_p2,raykeyframe);

    float d_z_d_z0 = dot(g_d_N_d_z0,g_pr_p0)/N_p0_dot_ray - g_N_p0_dot_point*dot(g_d_N_d_z0,raykeyframe)/(N_p0_dot_ray*N_p0_dot_ray);
    float d_z_d_z1 = dot(g_d_N_d_z1,g_pr_p1)/N_p1_dot_ray - g_N_p1_dot_point*dot(g_d_N_d_z1,raykeyframe)/(N_p1_dot_ray*N_p1_dot_ray);
    float d_z_d_z2 = dot(g_d_N_d_z2,g_pr_p2)/N_p2_dot_ray - g_N_p2_dot_point*dot(g_d_N_d_z2,raykeyframe)/(N_p2_dot_ray*N_p2_dot_ray);

    float d_I_d_z0 = d_I_d_z*d_z_d_z0;
    float d_I_d_z1 = d_I_d_z*d_z_d_z1;
    float d_I_d_z2 = d_I_d_z*d_z_d_z2;

    float error = iframe - ikeyframe;

    f_vertexID = vec3(g_vertexID[0], g_vertexID[1], g_vertexID[2]);
    f_primitiveID = gl_PrimitiveID;
    f_error = error;
    f_d_I_d_z0 = d_I_d_z0;
    f_d_I_d_z1 = d_I_d_z1;
    f_d_I_d_z2 = d_I_d_z2;
}

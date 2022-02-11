#version 330 core
layout(location = 0) out vec3 f_vertexID;
layout(location = 1) out float f_primitiveID;
layout(location = 2) out float f_error;
layout(location = 3) out float f_d_I_d_z0;
layout(location = 4) out float f_d_I_d_z1;
layout(location = 5) out float f_d_I_d_z2;

in vec3 g_pframe;
in vec3 g_pkeyframe;
in vec2 g_u;

flat in float g_idepth[3];
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

uniform int srclvl;

// texture samplers
uniform sampler2D keyframe;
uniform sampler2D frame;
uniform sampler2D frameDer;

void main()
{
    //if(g_u.x < 0.05/dx || g_u.x > 0.95/dx || g_u.y < 0.05/dy || g_u.y > 0.95/dy)
    //    discard;

    //from frame perspective
    vec2 ukeyframe = vec2(g_u.x,1.0/dy-g_u.y);
    vec3 pkeyframe = g_pkeyframe;
    float keyframeDepth = g_pkeyframe.z;
    vec2 uframe  = vec2(gl_FragCoord.x,gl_FragCoord.y);
    vec3 pframe = (framePose*vec4(pkeyframe,1.0)).xyz;
    float frameDepth = gl_FragCoord.z/gl_FragCoord.w;

    //from keyframe perspective
    //vec2 uframe  = vec2(g_u.x,1.0/dy-g_u.y);
    //vec2 ukeyframe = vec2(gl_FragCoord.x,gl_FragCoord.y);
    //float keyframeDepth = gl_FragCoord.z/gl_FragCoord.w;

    vec2 ukeyframeTexCoord = vec2(ukeyframe.x*dx, ukeyframe.y*dy);
    vec2 uframeTexCoord = vec2(uframe.x*dx, uframe.y*dy);

    float ikeyframe = textureLod(keyframe,ukeyframeTexCoord,srclvl).x*255.0;
    //vec2 dkeyframe = texture(keyframeDer,ukeyframeTexCoord).xy;

    float iframe = textureLod(frame,uframeTexCoord,srclvl).x*255.0;
    vec2 dframe = textureLod(frameDer,uframeTexCoord,srclvl).xy;

    vec3 d_I_d_pframe = vec3(0);
    d_I_d_pframe.x = dframe.x*fx/frameDepth;
    d_I_d_pframe.y = dframe.y*fy/frameDepth;
    d_I_d_pframe.z = -(d_I_d_pframe.x*pframe.x/frameDepth + d_I_d_pframe.y*pframe.y/frameDepth);

    vec3 raykeyframe = pkeyframe/keyframeDepth;

    vec3 d_pframe_d_z = mat3(framePose)*raykeyframe;

    float d_I_d_z = dot(d_I_d_pframe,d_pframe_d_z);

    float N_p0_dot_ray = dot(g_N_p0,raykeyframe);
    float N_p1_dot_ray = dot(g_N_p1,raykeyframe);
    float N_p2_dot_ray = dot(g_N_p2,raykeyframe);

    float d_z_d_z0 = dot(g_d_N_d_z0,g_pr_p0)/N_p0_dot_ray - g_N_p0_dot_point*dot(g_d_N_d_z0,raykeyframe)/(N_p0_dot_ray*N_p0_dot_ray);
    float d_z_d_z1 = dot(g_d_N_d_z1,g_pr_p1)/N_p1_dot_ray - g_N_p1_dot_point*dot(g_d_N_d_z1,raykeyframe)/(N_p1_dot_ray*N_p1_dot_ray);
    float d_z_d_z2 = dot(g_d_N_d_z2,g_pr_p2)/N_p2_dot_ray - g_N_p2_dot_point*dot(g_d_N_d_z2,raykeyframe)/(N_p2_dot_ray*N_p2_dot_ray);

    float d_z0_d_iz0 = -1.0/(g_idepth[0]*g_idepth[0]);
    float d_z1_d_iz1 = -1.0/(g_idepth[1]*g_idepth[1]);
    float d_z2_d_iz2 = -1.0/(g_idepth[2]*g_idepth[2]);

    float d_I_d_z0 = d_I_d_z*d_z_d_z0*d_z0_d_iz0;
    float d_I_d_z1 = d_I_d_z*d_z_d_z1*d_z1_d_iz1;
    float d_I_d_z2 = d_I_d_z*d_z_d_z2*d_z2_d_iz2;

    float error = iframe - ikeyframe;

    f_vertexID = vec3(g_vertexID[0], g_vertexID[1], g_vertexID[2]);
    f_primitiveID = 1.0/pframe.z - abs(error)/127.0;//d_z_d_z0;// gl_PrimitiveID;
    f_error = error;
    f_d_I_d_z0 = d_I_d_z0;
    f_d_I_d_z1 = d_I_d_z1;
    f_d_I_d_z2 = d_I_d_z2;
}

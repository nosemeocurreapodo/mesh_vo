#version 330 core
layout(location = 0) out ivec4 f_vertexID;
layout(location = 1) out int f_primitiveID;
layout(location = 2) out float f_error;
layout(location = 3) out vec4 f_d_I_d_p0;
layout(location = 4) out vec4 f_d_I_d_p1;
layout(location = 5) out vec4 f_d_I_d_p2;

in vec2 g_ukf;
in vec2 g_uf;
in float g_depth;
in vec3 g_pw; // depende de depth, depth no es lineal

flat in mat3 g_KRK_1;

flat in int g_vertexID[3];
flat in vec3 g_normal;

flat in vec3 g_d_N_d_p0[3];
flat in vec3 g_d_N_d_p1[3];
flat in vec3 g_d_N_d_p2[3];
flat in vec3 g_p1[3];

uniform float width;
uniform float height;

uniform mat3 K;
uniform mat3 invK;

// texture samplers
uniform sampler2D keyframe;
uniform sampler2D frame;
uniform sampler2D frameDerivative;

void main()
{
    vec2 ukfTexCoord = vec2(g_ukeyframe.x/width, g_ukeyframe.y/height);
    vec2 ufTexCoord = vec2(g_uframe.x/width, g_uframe.y/height);

    float NtP = dot(g_normal, g_pw/g_pw.z);

    float kf = texture(keyframe,vec2(ukfTexCoord.x, 1.0 - ukfTexCoord.y)).r;
    float f = texture(frame,vec2(ufTexCoord.x, 1.0 - ufTexCoord.y)).r;

    //frameDerivative evaluada en up (vector2d)
    vec2 d_I_d_uf = texture(frameDerivative, vec2(ufTexCoord.x, 1.0 - ufTexCoord.y)).rg;//compensar que frame va de 0 - 1

    //d_uf_d_pk (matriz 2x3) evaluada en up, z
    mat3 d_uf_d_pk;
    d_uf_d_pk[0] = vec3(1.0,0.0,0.0);
    d_uf_d_pk[1] = vec3(0.0,1.0,0.0);
    d_uf_d_pk[2] = vec3(-g_uf,0.0);
    d_uf_d_pk = d_uf_d_pk/g_pf.z;

mat3 rotframe = mat3(framePose);
mat3 rotkeyframe = mat3(keyframePose);

    vec3 d_pk_d_z = g_KRK_1*vec3(g_ukf,1.0f);

    //calculo d_uf_d_z
    vec2 d_uf_d_z = (d_uf_d_pk*g_d_pk_d_z).xy;

    //finalmente calculo y guardo d_f_d_z
    float d_I_d_z = dot(d_f_d_uf, d_uf_d_z);// (d_f_d_uf.x*d_uf_d_z.x + d_f_d_uf.y*d_uf_d_z.y);
    float error = (f - kf);

    vec3 pw1mp_0 = g_p1[0]-pw;
    vec3 pw1mp_1 = g_p1[1]-pw;
    vec3 pw1mp_2 = g_p1[2]-pw;
    //vec3 d_z_d_p0 = vec3(dot(g_d_N_d_p0[0],pw1mp_0),dot(g_d_N_d_p0[1],pw1mp_0),dot(g_d_N_d_p0[2],pw1mp_0))/NtP;
    //vec3 d_z_d_p1 = vec3(dot(g_d_N_d_p1[0],pw1mp_1),dot(g_d_N_d_p1[1],pw1mp_1),dot(g_d_N_d_p1[2],pw1mp_1))/NtP;
    //vec3 d_z_d_p2 = vec3(dot(g_d_N_d_p2[0],pw1mp_2),dot(g_d_N_d_p2[1],pw1mp_2),dot(g_d_N_d_p2[2],pw1mp_2))/NtP;
    vec3 d_z_d_p0 = vec3(0.0,0.0,dot(g_d_N_d_p0[2],pw1mp_0))/NtP;
    vec3 d_z_d_p1 = vec3(0.0,0.0,dot(g_d_N_d_p1[2],pw1mp_1))/NtP;
    vec3 d_z_d_p2 = vec3(0.0,0.0,dot(g_d_N_d_p2[2],pw1mp_2))/NtP;

    vec3 d_I_d_p0 = d_I_d_z*d_z_d_p0;
    vec3 d_I_d_p1 = d_I_d_z*d_z_d_p1;
    vec3 d_I_d_p2 = d_I_d_z*d_z_d_p2;

    f_vertexID = ivec4(g_vertexID[0], g_vertexID[1], g_vertexID[2], 1);
    f_primitiveID = gl_PrimitiveID;
    f_error = error;
    f_d_I_d_p0 = vec4(d_I_d_p0, 1.0);
    f_d_I_d_p1 = vec4(d_I_d_p1, 1.0);
    f_d_I_d_p2 = vec4(d_I_d_p2, 1.0);
}

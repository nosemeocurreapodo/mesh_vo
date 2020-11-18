#version 330 core
layout(location = 0) out vec4 f_debugView;
layout(location = 1) out ivec4 f_vertexID;
layout(location = 2) out int f_primitiveID;
layout(location = 3) out float f_error;
layout(location = 4) out vec4 f_d_I_d_p0;
layout(location = 5) out vec4 f_d_I_d_p1;
layout(location = 6) out vec4 f_d_I_d_p2;

in vec2 g_u;
in vec2 g_uTexCoord;
in float g_depth;
in vec3 g_pw; // depende de depth, depth no es lineal
in vec3 g_pk;
in vec3 g_d_pk_d_z;

flat in int g_vertexID[3];
flat in vec3 g_normal;
flat in vec3 g_d_N_d_p0[3];
flat in vec3 g_d_N_d_p1[3];
flat in vec3 g_d_N_d_p2[3];
flat in vec3 g_x1[3];

flat in int isInitialized;

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
    if(isInitialized == 0)
        discard;

    vec2 ukf = g_u;
    vec2 uf = g_pk.xy/g_pk.z;
    vec3 pw = g_pw;

    float kf_depth = g_depth;
    float f_depth = g_pk.z;

    vec2 ukfTexCoord = g_uTexCoord;
    vec2 ufTexCoord = vec2(uf.x/width, uf.y/height);

    float NtP = dot(g_normal, pw/pw.z);
    float angle = NtP/(length(g_normal)*length(pw/pw.z));
    if(abs(angle) < 0.01)
        discard;

    float kf = texture(keyframe,vec2(ukfTexCoord.x, 1.0 - ukfTexCoord.y)).r;
    float f = texture(frame,vec2(ufTexCoord.x, ufTexCoord.y)).r;

    vec2 d_f_d_uf = vec2(0.0,0.0);
    vec2 d_uf_d_z = vec2(0.0,0.0);
    float d_I_d_z = 0.0;
    float error = 0.0;
    vec3 d_I_d_p0 = vec3(0.0,0.0,0.0);
    vec3 d_I_d_p1 = vec3(0.0,0.0,0.0);
    vec3 d_I_d_p2 = vec3(0.0,0.0,0.0);
    vec3 d_z_d_p0 = vec3(0.0,0.0,0.0);
    vec3 d_z_d_p1 = vec3(0.0,0.0,0.0);
    vec3 d_z_d_p2 = vec3(0.0,0.0,0.0);
    vec3 pw1mp_0 = vec3(0.0,0.0,0.0);
    vec3 pw1mp_1 = vec3(0.0,0.0,0.0);
    vec3 pw1mp_2 = vec3(0.0,0.0,0.0);

    if(kf >= 0.0 && f >= 0.0)
    {
        //frameDerivative evaluada en up (vector2d)
        d_f_d_uf = texture(frameDerivative, vec2(ufTexCoord.x, 1.0 - ufTexCoord.y)).rg;//compensar que frame va de 0 - 1

        //d_uf_d_pk (matriz 2x3) evaluada en up, z
        mat3 d_uf_d_pk;
        d_uf_d_pk[0] = vec3(1.0,0.0,0.0);
        d_uf_d_pk[1] = vec3(0.0,1.0,0.0);
        d_uf_d_pk[2] = vec3(-uf,0.0);
        d_uf_d_pk = d_uf_d_pk/f_depth;

        //calculo d_uf_d_z
        d_uf_d_z = (d_uf_d_pk*g_d_pk_d_z).xy;

        //finalmente calculo y guardo d_f_d_z
        d_I_d_z = dot(d_f_d_uf, d_uf_d_z);// (d_f_d_uf.x*d_uf_d_z.x + d_f_d_uf.y*d_uf_d_z.y);
        error = (f - kf);
    }

    if(d_I_d_z != 0.0)
    {
        pw1mp_0 = g_x1[0]-pw;
        pw1mp_1 = g_x1[1]-pw;
        pw1mp_2 = g_x1[2]-pw;
        //vec3 d_z_d_p0 = vec3(dot(g_d_N_d_p0[0],pw1mp_0),dot(g_d_N_d_p0[1],pw1mp_0),dot(g_d_N_d_p0[2],pw1mp_0))/NtP;
        //vec3 d_z_d_p1 = vec3(dot(g_d_N_d_p1[0],pw1mp_1),dot(g_d_N_d_p1[1],pw1mp_1),dot(g_d_N_d_p1[2],pw1mp_1))/NtP;
        //vec3 d_z_d_p2 = vec3(dot(g_d_N_d_p2[0],pw1mp_2),dot(g_d_N_d_p2[1],pw1mp_2),dot(g_d_N_d_p2[2],pw1mp_2))/NtP;
        d_z_d_p0 = vec3(0.0,0.0,dot(g_d_N_d_p0[2],pw1mp_0))/NtP;
        d_z_d_p1 = vec3(0.0,0.0,dot(g_d_N_d_p1[2],pw1mp_1))/NtP;
        d_z_d_p2 = vec3(0.0,0.0,dot(g_d_N_d_p2[2],pw1mp_2))/NtP;

        d_I_d_p0 = d_I_d_z*d_z_d_p0;
        d_I_d_p1 = d_I_d_z*d_z_d_p1;
        d_I_d_p2 = d_I_d_z*d_z_d_p2;
    }

    f_vertexID = ivec4(g_vertexID[0], g_vertexID[1], g_vertexID[2], 1);
    f_primitiveID = gl_PrimitiveID;
    f_error = error;
    f_d_I_d_p0 = vec4(d_I_d_p0, 1.0);
    f_d_I_d_p1 = vec4(d_I_d_p1, 1.0);
    f_d_I_d_p2 = vec4(d_I_d_p2, 1.0);

    //f_debugView = vec4(d_f_d_uf.x/1.0+0.5, d_f_d_uf.y/1.0+0.5, 0.0, 1.0);
    //f_debugView = vec4(d_uf_d_z.x/40.0 + 0.5, d_uf_d_z.y/40.0 + 0.5, 0.0, 1.0);
    //f_debugView = vec4(d_I_d_z/20.0 + 0.5, d_I_d_z/20.0 + 0.5, d_I_d_z/20.0 + 0.5, 1.0);
    //f_debugView = vec4(d_z_d_p0.z/20.0 + 0.5, d_z_d_p1.z/20.0 + 0.5, d_z_d_p2.z/20.0 + 0.5, 1.0);
    //f_debugView = vec4(pw1mp_0.z/2.0 + 0.5, pw1mp_1.z/2.0 + 0.5, pw1mp_2.z/2.0 + 0.5, 1.0);
    //f_debugView = vec4(g_x1[0].x/2.0 + 0.5, g_x1[0].y/2.0 + 0.5, g_x1[0].z/2.0 + 0.5, 1.0);
    //f_debugView = vec4(pw.x/2.0 + 0.5, pw.y/2.0 + 0.5, pw.z/2.0 + 0.5, 1.0);
    f_debugView = vec4(d_I_d_p0.x/2.0 + 0.5, d_I_d_p0.y/2.0 + 0.5, d_I_d_p0.z/2.0 + 0.5, 1.0);
    //f_debugView = vec4(error/2.0+0.5, error/2.0+0.5, error/2.0+0.5, 1.0);
    //f_debugView = vec4(kf, kf, kf, 1.0);
    //f_debugView = vec4(f, f, f, 1.0);
    //f_debugView = vec4(f_depth/10.0, f_depth/10.0, f_depth/10.0, 1.0);
}

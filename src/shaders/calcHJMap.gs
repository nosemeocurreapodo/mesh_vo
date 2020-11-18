#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec2 v_u[];
in vec2 v_uTexCoord[];
in float v_depth[];
in vec3 v_pw[];
in vec3 v_pk[];
in vec3 v_d_pk_d_z[];
flat in int v_vertexID[];

out vec2 g_u;
out vec2 g_uTexCoord; //lineal en la imagen
out float g_depth;
out vec3 g_pw; // depende de depth, depth no es lineal
out vec3 g_pk;
out vec3 g_d_pk_d_z;

flat out vec3 g_normal;
flat out int g_vertexID[3];

flat out vec3 g_d_N_d_p0[3];
flat out vec3 g_d_N_d_p1[3];
flat out vec3 g_d_N_d_p2[3];
flat out vec3 g_x1[3];

flat out int isInitialized;

uniform mat3 invK;
uniform float width;
uniform float height;

bool isOOB(vec2 u)
{
    if(u.x < 0.0 || u.y < 0.0 || u.x > width || u.y > height)
        return true;
    else
        return false;
}

void main() {

    gl_PrimitiveID = gl_PrimitiveIDIn;
    vec3 normal = cross(v_pw[1] - v_pw[0], v_pw[2] - v_pw[0]);

    vec2 uf[3];
    uf[0] = v_pk[0].xy/v_pk[0].z;
    uf[1] = v_pk[1].xy/v_pk[1].z;
    uf[2] = v_pk[2].xy/v_pk[2].z;

    if(v_depth[0] > 0.0 && v_depth[1] > 0.0 && v_depth[2] > 0.0 )
        isInitialized = 1;
    else
        isInitialized = 0;

    if(isOOB(uf[0]) || isOOB(uf[1]) || isOOB(uf[2]))
        isInitialized = 0;

    vec3 pw2mpw1;
    //p = u, v, idepth
    pw2mpw1 = (v_pw[2]-v_pw[1]);
    g_d_N_d_p0[0] = vec3(0.0,pw2mpw1.z*invK[0][0]*v_depth[0],-invK[0][0]*pw2mpw1.y*v_depth[0]);
    g_d_N_d_p0[1] = vec3(-invK[1][1]*pw2mpw1.z*v_depth[0],0.0,pw2mpw1.x*invK[1][1]*v_depth[0]);
    g_d_N_d_p0[2] = cross(pw2mpw1, -invK*vec3(v_u[0],1.0)*(v_depth[0]*v_depth[0]));//invK*vec3(v_u[0],1.0))/normal_length;// v_pw[0]/v_pw[0].z);
    g_x1[0] = v_pw[1];

    pw2mpw1 = (v_pw[0]-v_pw[2]);
    g_d_N_d_p1[0] = vec3(0.0,pw2mpw1.z*invK[0][0]*v_depth[1],-invK[0][0]*pw2mpw1.y*v_depth[1]);
    g_d_N_d_p1[1] = vec3(-invK[1][1]*pw2mpw1.z*v_depth[1],0.0,pw2mpw1.x*invK[1][1]*v_depth[1]);
    g_d_N_d_p1[2] = cross(pw2mpw1, -invK*vec3(v_u[1],1.0)*(v_depth[1]*v_depth[1]));//invK*vec3(v_u[1],1.0))/normal_length;//v_pw[1]/v_pw[1].z);
    g_x1[1] = v_pw[2];

    pw2mpw1 = (v_pw[1]-v_pw[0]);
    g_d_N_d_p2[0] = vec3(0.0,pw2mpw1.z*invK[0][0]*v_depth[2],-invK[0][0]*pw2mpw1.y*v_depth[2]);
    g_d_N_d_p2[1] = vec3(-invK[1][1]*pw2mpw1.z*v_depth[2],0.0,pw2mpw1.x*invK[1][1]*v_depth[2]);
    g_d_N_d_p2[2] = cross(pw2mpw1, -invK*vec3(v_u[2],1.0)*(v_depth[2]*v_depth[2]));
    g_x1[2] = v_pw[0];


    g_normal = normal;///length(normal);
    // vertice 0
    g_vertexID[0] = v_vertexID[0];
    g_u = v_u[0];
    g_uTexCoord = v_uTexCoord[0];
    g_depth = v_depth[0];
    g_pw = v_pw[0];
    g_pk = v_pk[0];
    g_d_pk_d_z = v_d_pk_d_z[0];

    gl_Position = gl_in[0].gl_Position;
    EmitVertex();

    //vertice 1
    g_vertexID[1] = v_vertexID[1];
    g_u = v_u[1];
    g_uTexCoord = v_uTexCoord[1];
    g_depth = v_depth[1];
    g_pw = v_pw[1];
    g_pk = v_pk[1];
    g_d_pk_d_z = v_d_pk_d_z[1];

    gl_Position = gl_in[1].gl_Position;
    EmitVertex();

    //vertice 2
    g_vertexID[2] = v_vertexID[2];
    g_u = v_u[2];
    g_uTexCoord = v_uTexCoord[2];
    g_depth = v_depth[2];
    g_pw = v_pw[2];
    g_pk = v_pk[2];
    g_d_pk_d_z = v_d_pk_d_z[2];

    gl_Position = gl_in[2].gl_Position;
    EmitVertex();

    EndPrimitive();
}

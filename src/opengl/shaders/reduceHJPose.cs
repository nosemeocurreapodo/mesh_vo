#version 430 core

// inputs from compute shader
//
// in uvec3 gl_NumWorkGroups;
// in uvec3 gl_WorkGroupID;
// in uvec3 gl_LocalInvocationID;
// in uvec3 gl_GlobalInvocationID;
// in uint gl_LocalInvocationIndex;
//
// more details at https://www.khronos.org/opengl/wiki/Compute_Shader#Inputs

// outputs will need to be either in a shader storage buffer object
// or an image load store
//
// more details at https://www.khronos.org/opengl/wiki/Compute_Shader#Outputs

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(rgba32f, binding = 0) uniform image2D traTex;
layout(rgba32f, binding = 1) uniform image2D rotTex;
layout(r32f, binding = 2) uniform image2D resTex;

layout(rgba32f, binding = 3) uniform image2D gradient1Tex;
layout(rgba32f, binding = 4) uniform image2D gradient2Tex;
layout(rgba32f, binding = 5) uniform image2D hessian1Tex;
layout(rgba32f, binding = 6) uniform image2D hessian2Tex;
layout(rgba32f, binding = 7) uniform image2D hessian3Tex;
layout(rgba32f, binding = 8) uniform image2D hessian4Tex;
layout(rgba32f, binding = 9) uniform image2D hessian5Tex;
layout(rgba32f, binding = 10) uniform image2D hessian6Tex;
layout(r32f, binding = 11) uniform image2D errorTex;
layout(r32f, binding = 12) uniform image2D countTex;

shared float J_pose[6][gl_WorkGroupSize.x*gl_WorkGroupSize.y];
shared float H_pose[21][gl_WorkGroupSize.x*gl_WorkGroupSize.y];
shared float error[gl_WorkGroupSize.x*gl_WorkGroupSize.y];
shared float count[gl_WorkGroupSize.x*gl_WorkGroupSize.y];

void main() {

    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    vec3 tra  = imageLoad(traTex, pos).xyz;
    vec3 rot  = imageLoad(rotTex, pos).xyz;
    float res = imageLoad(resTex, pos).x;

    float J[6];
    float c;
    if(res == 0.0)
    {
        tra.x = 0.0;
        tra.y = 0.0;
        tra.z = 0.0;
        rot.x = 0.0;
        rot.y = 0.0;
        rot.z = 0.0;

        c = 0.0;
    }
    else
    {
        tra.x = tra.x;
        tra.y = tra.y;
        tra.z = tra.z;
        rot.x = rot.x;
        rot.y = rot.y;
        rot.z = rot.z;

        c = 1.0;
    }

    J_pose[0][gl_LocalInvocationIndex] = tra.x*res;
    J_pose[1][gl_LocalInvocationIndex] = tra.y*res;
    J_pose[2][gl_LocalInvocationIndex] = tra.z*res;
    J_pose[3][gl_LocalInvocationIndex] = rot.x*res;
    J_pose[4][gl_LocalInvocationIndex] = rot.y*res;
    J_pose[5][gl_LocalInvocationIndex] = rot.z*res;

    H_pose[0][gl_LocalInvocationIndex] = tra.x*tra.x;
    H_pose[1][gl_LocalInvocationIndex] = tra.x*tra.y;
    H_pose[2][gl_LocalInvocationIndex] = tra.x*tra.z;
    H_pose[3][gl_LocalInvocationIndex] = tra.x*rot.x;
    H_pose[4][gl_LocalInvocationIndex] = tra.x*rot.y;
    H_pose[5][gl_LocalInvocationIndex] = tra.x*rot.z;

    H_pose[6][gl_LocalInvocationIndex] = tra.y*tra.y;
    H_pose[7][gl_LocalInvocationIndex] = tra.y*tra.z;
    H_pose[8][gl_LocalInvocationIndex] = tra.y*rot.x;
    H_pose[9][gl_LocalInvocationIndex] = tra.y*rot.y;
    H_pose[10][gl_LocalInvocationIndex] = tra.y*rot.z;

    H_pose[11][gl_LocalInvocationIndex] = tra.z*tra.z;
    H_pose[12][gl_LocalInvocationIndex] = tra.z*rot.x;
    H_pose[13][gl_LocalInvocationIndex] = tra.z*rot.y;
    H_pose[14][gl_LocalInvocationIndex] = tra.z*rot.z;

    H_pose[15][gl_LocalInvocationIndex] = rot.x*rot.x;
    H_pose[16][gl_LocalInvocationIndex] = rot.x*rot.y;
    H_pose[17][gl_LocalInvocationIndex] = rot.x*rot.z;

    H_pose[18][gl_LocalInvocationIndex] = rot.y*rot.y;
    H_pose[19][gl_LocalInvocationIndex] = rot.y*rot.z;

    H_pose[20][gl_LocalInvocationIndex] = rot.z*rot.z;

    error[gl_LocalInvocationIndex] = res*res;
    count[gl_LocalInvocationIndex] = c;

    for(uint stride = gl_WorkGroupSize.x*gl_WorkGroupSize.y/2; stride > 0; stride /=2)
    {
        barrier();
        if(gl_LocalInvocationIndex < stride)
        {
            error[gl_LocalInvocationIndex] += error[gl_LocalInvocationIndex + stride];
            count[gl_LocalInvocationIndex] += count[gl_LocalInvocationIndex + stride];

            for(int i  = 0; i < 6; i++)
            {
                J_pose[i][gl_LocalInvocationIndex] += J_pose[i][gl_LocalInvocationIndex + stride];
            }
            for(int i = 0; i < 21; i++)
            {
                H_pose[i][gl_LocalInvocationIndex] += H_pose[i][gl_LocalInvocationIndex + stride];
            }
        }
    }

    if(gl_LocalInvocationIndex == 0)
    {
        imageStore(gradient1Tex, ivec2(gl_WorkGroupID.xy), vec4(J_pose[0][0],J_pose[1][0],J_pose[2][0],0.0));
        imageStore(gradient2Tex, ivec2(gl_WorkGroupID.xy), vec4(J_pose[3][0],J_pose[4][0],J_pose[5][0],0.0));

        imageStore(hessian1Tex, ivec2(gl_WorkGroupID.xy), vec4(H_pose[0][0],H_pose[1][0],H_pose[2][0],H_pose[3][0]));
        imageStore(hessian2Tex, ivec2(gl_WorkGroupID.xy), vec4(H_pose[4][0],H_pose[5][0],H_pose[6][0],H_pose[7][0]));
        imageStore(hessian3Tex, ivec2(gl_WorkGroupID.xy), vec4(H_pose[8][0],H_pose[9][0],H_pose[10][0],H_pose[11][0]));
        imageStore(hessian4Tex, ivec2(gl_WorkGroupID.xy), vec4(H_pose[12][0],H_pose[13][0],H_pose[14][0],H_pose[15][0]));
        imageStore(hessian5Tex, ivec2(gl_WorkGroupID.xy), vec4(H_pose[16][0],H_pose[17][0],H_pose[18][0],H_pose[19][0]));
        imageStore(hessian6Tex, ivec2(gl_WorkGroupID.xy), vec4(H_pose[20][0],0.0,0.0,0.0));

        imageStore(errorTex, ivec2(gl_WorkGroupID.xy), vec4(error[0],0.0,0.0,0.0));
        imageStore(countTex, ivec2(gl_WorkGroupID.xy), vec4(count[0],0.0,0.0,0.0));
    }
}


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

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(r32f, binding = 0) uniform image2D traTex;
layout(r32f, binding = 1) uniform image2D rotTex;
layout(r32f, binding = 2) uniform image2D resTex;

shared float JShared[6][gl_WorkGroupSize.x*gl_WorkGroupSize.y];
shared float auxShared[gl_WorkGroupSize.x*gl_WorkGroupSize.y];
shared float resShared[gl_WorkGroupSize.x*gl_WorkGroupSize.y];
shared float J_pose[6];
shared float H_pose[21];

void main() {

    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    vec3 tra  = imageLoad(traTex, pos).xyz;
    vec3 rot  = imageLoad(rotTex, pos).xyz;
    float res = imageLoad(resTex, pos).x;

    if(res == 0.0)
        return;

    JShared[gl_LocalInvocationIndex][0] = tra.x;
    JShared[gl_LocalInvocationIndex][1] = tra.y;
    JShared[gl_LocalInvocationIndex][2] = tra.z;
    JShared[gl_LocalInvocationIndex][3] = rot.x;
    JShared[gl_LocalInvocationIndex][4] = rot.y;
    JShared[gl_LocalInvocationIndex][5] = rot.z;
    resShared[gl_LocalInvocationIndex] = res;

    barrier();

    for(int i = 0; i < 6; i++)
    {
        auxShared[gl_LocalInvocationIndex] = JShared[gl_LocalInvocationIndex][i]*resShared[gl_LocalInvocationIndex];

        for(uint stride = gl_WorkGroupSize.x*gl_WorkGroupSize.y/2; stride > 0; stride /=2)
        {
            barrier();
            if(gl_LocalInvocationIndex < stride)
            {
                auxShared[gl_LocalInvocationIndex] += auxShared[gl_LocalInvocationIndex + stride];
            }
        }

        if(gl_LocalInvocationIndex == 0)
            J_pose[i] = auxShared[gl_LocalInvocationIndex];
    }

    int index = 0;
    for(int i = 0; i < 6; i++)
    {
        for(int j = i; j < 6; j++)
        {
            auxShared[gl_LocalInvocationIndex] = JShared[gl_LocalInvocationIndex][i]*JShared[gl_LocalInvocationIndex][j];

            for(uint stride = gl_WorkGroupSize.x*gl_WorkGroupSize.y/2; stride > 0; stride /=2)
            {
                barrier();
                if(gl_LocalInvocationIndex < stride)
                {
                    auxShared[gl_LocalInvocationIndex] += auxShared[gl_LocalInvocationIndex + stride];
                }
            }

            if(gl_LocalInvocationIndex == 0)
                H_pose[index] = auxShared[gl_LocalInvocationIndex];
            index++;
        }
    }

    imageStore( out_tex, pos, vec4( in_val + 1, 0.0, 0.0, 0.0 ) );
}

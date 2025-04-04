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
layout(r32f, binding = 0) uniform image2D errorInTex;
layout(r32f, binding = 1) uniform image2D errorOutTex;
layout(r32f, binding = 2) uniform image2D countOutTex;

shared float errorLocal[gl_WorkGroupSize.x*gl_WorkGroupSize.y];
shared int countLocal[gl_WorkGroupSize.x*gl_WorkGroupSize.y];

void main()
{
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    float error = imageLoad(errorInTex, pos).r;
    if(error < 0.0)
    {
        errorLocal[gl_LocalInvocationIndex] = 0.0;
        countLocal[gl_LocalInvocationIndex] = 0;
    }
    else
    {
        errorLocal[gl_LocalInvocationIndex] = error;
        countLocal[gl_LocalInvocationIndex] = 1;
    }
    for(uint stride = gl_WorkGroupSize.x*gl_WorkGroupSize.y/2; stride > 0; stride /=2)
    {
        barrier();
        if(gl_LocalInvocationIndex < stride)
        {
            errorLocal[gl_LocalInvocationIndex] += errorLocal[gl_LocalInvocationIndex + stride];
            countLocal[gl_LocalInvocationIndex] += countLocal[gl_LocalInvocationIndex + stride];
        }
    }
    if(gl_LocalInvocationIndex == 0)
    {
        imageStore(errorOutTex, ivec2(gl_WorkGroupID.xy), vec4(errorLocal[0],0.0,0.0,0.0));
        imageStore(countOutTex, ivec2(gl_WorkGroupID.xy), vec4(float(countLocal[0]),0,0,0));
        //imageStore(countOutTex, ivec2(gl_WorkGroupID.xy), vec4(10,0,0,0));
        //imageStore(errorOutTex, ivec2(gl_WorkGroupID.xy), vec4(2.345,0.0,0.0,0.0));
    }
}

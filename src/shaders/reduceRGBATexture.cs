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

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
layout(rgba32f, binding = 0) uniform image2D inTex;
layout(rgba32f, binding = 1) uniform image2D outTex;

shared vec4 sharedData[gl_WorkGroupSize.x*gl_WorkGroupSize.y];

void main() {

    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

    vec4 data  = imageLoad(inTex, pos).xyzw;

    sharedData[gl_LocalInvocationIndex] = data;

    for(uint stride = gl_WorkGroupSize.x*gl_WorkGroupSize.y/2; stride > 0; stride /=2)
    {
        barrier();
        if(gl_LocalInvocationIndex < stride)
        {
            sharedData[gl_LocalInvocationIndex] += sharedData[gl_LocalInvocationIndex + stride];
        }
    }

    if(gl_LocalInvocationIndex == 0)
    {
        imageStore(outTex, ivec2(gl_WorkGroupID.xy), sharedData[0]);
    }
}

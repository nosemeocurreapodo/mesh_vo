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
layout(r8ui, binding = 0) uniform uimage2D keyframeTex;
layout(r8ui, binding = 1) uniform uimage2D frameTex;
layout(r32f, binding = 2) uniform image2D invDepthTex;
layout(r32f, binding = 3) uniform image2D errorTex;
layout(r32f, binding = 4) uniform image2D countTex;

uniform mat4 framePose;

uniform float fx;
uniform float fy;
uniform float cx;
uniform float cy;

uniform float fxinv;
uniform float fyinv;
uniform float cxinv;
uniform float cyinv;

uniform float dx;
uniform float dy;

shared float errorLocal[gl_WorkGroupSize.x*gl_WorkGroupSize.y];
shared int countLocal[gl_WorkGroupSize.x*gl_WorkGroupSize.y];

void main()
{
    bool isGood = true;

    vec2 kfpixel = vec2(gl_GlobalInvocationID.xy);
    float kfidepth = imageLoad(invDepthTex, ivec2(kfpixel.x,1.0/dy-kfpixel.y)).x;
    vec3 kfray = vec3(fxinv*kfpixel.x + cxinv, fyinv*kfpixel.y + cyinv, 1.0);
    vec3 kfpoint = kfray/kfidepth;

    vec3 fpoint = (framePose*vec4(kfpoint,1.0)).xyz;

    if(fpoint.z <= 0.0)
        isGood = false;

    vec3 fray = fpoint/fpoint.z;
    vec2 fpixel = vec2(fx*fray.x+cx,fy*fray.y+cy);

    if(fpixel.x < 0.0 || fpixel.x >= 1.0/dx || fpixel.y < 0.0 || fpixel.y >= 1.0/dy)
        isGood = false;

    float kf = imageLoad(keyframeTex, ivec2(kfpixel.x,1.0/dy-kfpixel.y)).x;
    float f = imageLoad(frameTex, ivec2(fpixel.x,1.0/dy-fpixel.y)).x;

    float error = pow(f - kf, 2.0);

    if(isGood == true)
    {
        errorLocal[gl_LocalInvocationIndex] = error;
        countLocal[gl_LocalInvocationIndex] = 1;
    }
    else
    {
        errorLocal[gl_LocalInvocationIndex] = 0.0;
        countLocal[gl_LocalInvocationIndex] = 0;
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
        imageStore(errorTex, ivec2(gl_WorkGroupID.xy), vec4(errorLocal[0],0.0,0.0,0.0));
        imageStore(countTex, ivec2(gl_WorkGroupID.xy), vec4(float(countLocal[0]),0,0,0));
        //imageStore(countOutTex, ivec2(gl_WorkGroupID.xy), vec4(10,0,0,0));
        //imageStore(errorOutTex, ivec2(gl_WorkGroupID.xy), vec4(2.345,0.0,0.0,0.0));
    }
}


#version 430 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(r8ui   , binding = 0 ) uniform uimage2D keyframeTex;
layout(r8ui   , binding = 1 ) uniform uimage2D frameTex;
layout(rg32f  , binding = 2 ) uniform image2D  frameDerTex;
layout(r32f   , binding = 3 ) uniform image2D  invDepthTex;

layout(rgba32f, binding = 4 ) uniform image2D  gradient1Tex;
layout(rgba32f, binding = 5 ) uniform image2D  gradient2Tex;
layout(rgba32f, binding = 6 ) uniform image2D  hessian1Tex;
layout(rgba32f, binding = 7 ) uniform image2D  hessian2Tex;
layout(rgba32f, binding = 8 ) uniform image2D  hessian3Tex;
layout(rgba32f, binding = 9 ) uniform image2D  hessian4Tex;
layout(rgba32f, binding = 10) uniform image2D  hessian5Tex;
layout(rgba32f, binding = 11) uniform image2D  hessian6Tex;
layout(r32f   , binding = 12) uniform image2D  errorTex;
layout(r32f   , binding = 13) uniform image2D  countTex;

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

shared float J_pose[6][gl_WorkGroupSize.x*gl_WorkGroupSize.y];
shared float H_pose[21][gl_WorkGroupSize.x*gl_WorkGroupSize.y];
shared float error[gl_WorkGroupSize.x*gl_WorkGroupSize.y];
shared float count[gl_WorkGroupSize.x*gl_WorkGroupSize.y];

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

    float c = 1.0;
    if(!isGood)
        c = 0.0;

    vec2 f_der = imageLoad(frameDerTex, ivec2(fpixel.x,1.0/dy-fpixel.y)).xy;

    float id = 1.0/fpoint.z;

    float v0 = f_der.x * fx * id;
    float v1 = f_der.y * fy * id;
    float v2 = -(v0 * fpoint.x + v1 * fpoint.y) * id;

    vec3 tra = vec3(v0, v1, v2);
    vec3 rot = vec3( -fpoint.z * v1 + fpoint.y * v2, fpoint.z * v0 - fpoint.x * v2, -fpoint.y * v0 + fpoint.x * v1);

    float kf = imageLoad(keyframeTex, ivec2(kfpixel.x,1.0/dy-kfpixel.y)).x;
    float f = imageLoad(frameTex, ivec2(fpixel.x,1.0/dy-fpixel.y)).x;

    float res = f - kf;

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

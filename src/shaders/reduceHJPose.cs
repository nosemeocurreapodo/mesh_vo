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

shared float J_pose[6][gl_WorkGroupSize.x*gl_WorkGroupSize.y];
shared float H_pose[21][gl_WorkGroupSize.x*gl_WorkGroupSize.y];

void main() {

    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    vec3 tra  = imageLoad(traTex, pos).xyz;
    vec3 rot  = imageLoad(rotTex, pos).xyz;
    float res = imageLoad(resTex, pos).x;

    if(res == 0.0)
        return;

    float J[6];
    J[0] = tra.x;
    J[1] = tra.y;
    J[2] = tra.z;
    J[3] = rot.x;
    J[4] = rot.y;
    J[5] = rot.z;

    J_pose[0][gl_LocalInvocationIndex] = J[0]*res;
    J_pose[1][gl_LocalInvocationIndex] = J[1]*res;
    J_pose[2][gl_LocalInvocationIndex] = J[2]*res;
    J_pose[3][gl_LocalInvocationIndex] = J[3]*res;
    J_pose[4][gl_LocalInvocationIndex] = J[4]*res;
    J_pose[5][gl_LocalInvocationIndex] = J[5]*res;

    H_pose[0][gl_LocalInvocationIndex] = J[0]*J[0];
    H_pose[1][gl_LocalInvocationIndex] = J[0]*J[1];
    H_pose[2][gl_LocalInvocationIndex] = J[0]*J[2];
    H_pose[3][gl_LocalInvocationIndex] = J[0]*J[3];
    H_pose[4][gl_LocalInvocationIndex] = J[0]*J[4];
    H_pose[5][gl_LocalInvocationIndex] = J[0]*J[5];

    H_pose[6][gl_LocalInvocationIndex] = J[1]*J[1];
    H_pose[7][gl_LocalInvocationIndex] = J[1]*J[2];
    H_pose[8][gl_LocalInvocationIndex] = J[1]*J[3];
    H_pose[9][gl_LocalInvocationIndex] = J[1]*J[4];
    H_pose[10][gl_LocalInvocationIndex] = J[1]*J[5];

    H_pose[11][gl_LocalInvocationIndex] = J[2]*J[2];
    H_pose[12][gl_LocalInvocationIndex] = J[2]*J[3];
    H_pose[13][gl_LocalInvocationIndex] = J[2]*J[4];
    H_pose[14][gl_LocalInvocationIndex] = J[2]*J[5];

    H_pose[15][gl_LocalInvocationIndex] = J[3]*J[3];
    H_pose[16][gl_LocalInvocationIndex] = J[3]*J[4];
    H_pose[17][gl_LocalInvocationIndex] = J[3]*J[5];

    H_pose[18][gl_LocalInvocationIndex] = J[4]*J[4];
    H_pose[19][gl_LocalInvocationIndex] = J[4]*J[5];

    H_pose[20][gl_LocalInvocationIndex] = J[5]*J[5];

    for(uint stride = gl_WorkGroupSize.x*gl_WorkGroupSize.y/2; stride > 0; stride /=2)
    {
        barrier();
        if(gl_LocalInvocationIndex < stride)
        {
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
    /*
    if(gl_LocalInvocationIndex == 0)
    {
        float result = -1.0;
        if(countLocal[0] > 0)
            result = errorLocal[0];
        imageStore(errorOutTex, ivec2(gl_WorkGroupID.xy), vec4(result,0.0,0.0,0.0));
        imageStore(countOutTex, ivec2(gl_WorkGroupID.xy), vec4(float(countLocal[0]),0,0,0));
        //imageStore(countOutTex, ivec2(gl_WorkGroupID.xy), vec4(10,0,0,0));
        //imageStore(errorOutTex, ivec2(gl_WorkGroupID.xy), vec4(2.345,0.0,0.0,0.0));
    }

    // store new value in image
    imageStore( out_tex, pos, vec4( in_val + 1, 0.0, 0.0, 0.0 ) );
    */
}

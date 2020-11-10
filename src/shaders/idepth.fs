#version 330 core
layout(location = 0) out float f_depth;

in float v_depth;

void main()
{
    f_depth = gl_FragCoord.w/gl_FragCoord.z;
    //f_depth = gl_FragCoord.w/gl_FragCoord.z - 1.0/v_depth;
}

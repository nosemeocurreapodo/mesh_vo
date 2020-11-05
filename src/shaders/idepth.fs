#version 330 core
layout(location = 0) out float f_depth;

in vec3 g_pw;
in float g_depth;
flat in vec3 g_normal;

flat in int isInitialized;

void main()
{
    //if(isInitialized == 0)
    //    discard;
    //float angle = dot(g_normal, g_pw)/(length(g_normal)*length(g_pw));
    //if(abs(angle) < 0.1)
    //    discard;
    f_depth = 1.0;// 1.0/g_depth;
    //f_state = 3; // depth ok, witch triangle it belongs to
}

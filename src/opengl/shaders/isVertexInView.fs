#version 330 core
layout(location = 0) out vec3 f_vertexID;

in vec3 g_pkeyframe;
flat in int g_vertexID[3];

uniform mat4 framePose;

void main()
{
    vec3 pkeyframe = g_pkeyframe;

    float cosangle = abs(dot(normalize(pkeyframe),normalize(framePose[3].xyz)));
    if(cosangle >= 0.8)
      discard;

    f_vertexID = vec3(g_vertexID[0], g_vertexID[1], g_vertexID[2]);
}

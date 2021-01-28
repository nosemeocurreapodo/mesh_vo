#version 330 core
layout(location = 0) out vec4 f_j1;
layout(location = 1) out vec4 f_j2;
layout(location = 2) out vec4 f_j3;
layout(location = 3) out vec4 f_j4;
layout(location = 4) out vec4 f_j5;
layout(location = 5) out vec4 f_j6;
layout(location = 6) out vec4 f_j7;

uniform sampler2D residualTexture;
uniform sampler2D traTexture;
uniform sampler2D rotTexture;

uniform int lvlin;
uniform int lvlout;

void main()
{
    int lvldiff = lvlout - lvlin;

    if(lvldiff < 0)
      discard;

    int window = int(pow(2,lvldiff));

    vec4 j1 = vec4(0.0);
    vec4 j2 = vec4(0.0);
    vec4 j3 = vec4(0.0);
    vec4 j4 = vec4(0.0);
    vec4 j5 = vec4(0.0);
    vec4 j6 = vec4(0.0);
    vec4 j7 = vec4(0.0);

    int count = 0;
    for(int y = 0; y < window; y+=1)
      for(int x = 0; x < window; x+=1)
      {
        ivec2 address = ivec2((gl_FragCoord.xy - vec2(0.5,0.5))*window)+ivec2(x,y);
        float res = texelFetch(residualTexture, address, lvlin).x;

        vec3 tra = texelFetch(traTexture, address, lvlin).xyz;
        vec3 rot = texelFetch(rotTexture, address, lvlin).xyz;

         j1 += vec4(tra.xyz,rot.x)*res;
         j2 += vec4(rot.yz*res,tra.x*tra.xy);
         j3 += vec4(tra.x*tra.z,tra.x*rot.xyz);
         j4 += vec4(tra.y*tra.yz,tra.y*rot.xy);
         j5 += vec4(tra.y*rot.z,tra.z*tra.z,tra.z*rot.xy);
         j6 += vec4(tra.z*rot.z,rot.x*rot.xyz);
         j7 += vec4(rot.y*rot.yz,rot.z*rot.z,0.0);

         count++;
      }

    /*
    if(count > 0)
    {
      j1 /= count;
      j2 /= count;
      j3 /= count;
      j4 /= count;
      j5 /= count;
      j6 /= count;
      j7 /= count;
    }
    */

    f_j1 = j1;
    f_j2 = j2;
    f_j3 = j3;
    f_j4 = j4;
    f_j5 = j5;
    f_j6 = j6;
    f_j7 = j7;
}

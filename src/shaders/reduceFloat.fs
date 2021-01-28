#version 330 core
layout(location = 0) out float f_reduced;

uniform sampler2D inTexture;

uniform int lvlin;
uniform int lvlout;

void main()
{
    int lvldiff = lvlout - lvlin;

    if(lvldiff < 0)
      discard;

    int window = int(pow(2,lvldiff));

    float rresidual = 0.0;
    int count = 0;
    for(int y = 0; y < window; y+=1)
      for(int x = 0; x < window; x+=1)
      {
        float res = texelFetch(inTexture, ivec2( (gl_FragCoord.xy - vec2(0.5,0.5))*window)+ivec2(x,y), lvlin).x;
        if(res > 0.0)
        {
          rresidual += res;
          count++;
        }
      }

    if(count > 0)
      rresidual = rresidual/count;

    f_reduced = rresidual;
}

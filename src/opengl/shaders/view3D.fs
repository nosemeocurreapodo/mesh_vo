#version 330 core
layout(location = 0) out vec4 FragColor;

in vec2 v_u;

uniform sampler2D keyframe;

uniform float dx;
uniform float dy;

void main()
{
    vec2 ukeyframe = vec2(v_u.x,1.0/dy-v_u.y);
    vec2 ukeyframeTexCoord = vec2(ukeyframe.x*dx, ukeyframe.y*dy);
    float kf_pixel = texture(keyframe, ukeyframeTexCoord).x;
    float idepth = gl_FragCoord.w/gl_FragCoord.z;
    //FragColor = vec4(kf_pixel,kf_pixel,kf_pixel,1.0);
    FragColor = vec4(idepth,idepth,idepth,1.0);
}

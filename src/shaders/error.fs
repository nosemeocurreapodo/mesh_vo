#version 330 core
layout(location = 0) out float f_residual;

in vec3 v_pkeyframe;

uniform sampler2D keyframe;
uniform sampler2D frame;

uniform float fx;
uniform float fy;
uniform float cx;
uniform float cy;

uniform float dx;
uniform float dy;

void main()
{
    vec2 ukeyframe = vec2(fx*v_pkeyframe.x/v_pkeyframe.z+cx,fy*v_pkeyframe.y/v_pkeyframe.z+cy);
    vec2 uframe = vec2(gl_FragCoord.x,1.0/dy-gl_FragCoord.y);

    vec2 ukeyframeTexCoord = vec2(ukeyframe.x*dx, 1.0-ukeyframe.y*dy);
    vec2 uframeTexCoord = vec2(uframe.x*dx, 1.0-uframe.y*dy);

    float f_pixel = texture(frame, uframeTexCoord).x;
    float kf_pixel = texture(keyframe, ukeyframeTexCoord).x;

    f_residual = pow(f_pixel-kf_pixel,2.0);
}

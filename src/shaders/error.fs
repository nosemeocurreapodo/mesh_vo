#version 330 core
layout(location = 0) out float f_residual;

in vec3 v_pframe;
in vec2 v_ukeyframe;

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

vec2 ukeyframe = v_ukeyframe;
vec2 uframe  = vec2(gl_FragCoord.x,gl_FragCoord.y);

vec2 ukeyframeTexCoord = vec2(ukeyframe.x*dx, 1.0-ukeyframe.y*dy);
vec2 uframeTexCoord = vec2(uframe.x*dx, uframe.y*dy);

float f_pixel = texture(frame, uframeTexCoord).x*255.0;
float kf_pixel = texture(keyframe, ukeyframeTexCoord).x*255.0;

if(v_pframe.z <= 0.0)
  f_residual =pow(1000.0,2.0);
else
  f_residual = pow((f_pixel-kf_pixel),2.0);
}

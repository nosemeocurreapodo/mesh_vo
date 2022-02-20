#version 330 core
layout(location = 0) out vec3 f_vertexID;
layout(location = 1) out float f_residual;

in vec2 g_u;
flat in int g_vertexID[3];

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
//if(v_pframe.z < 0.0)
//  discard;

//if(v_u.x < 0.0 || v_u.x > 1.0/dx || v_u.y < 0.0 || v_u.y > 1.0/dy)
//  discard;

//from frame perspective
vec2 ukeyframe = vec2(g_u.x,1.0/dy-g_u.y);
vec2 uframe  = vec2(gl_FragCoord.x,gl_FragCoord.y);

//from keyframe perspective
//vec2 uframe = vec2(v_u.x,1.0/dy-v_u.y);
//vec2 ukeyframe  = vec2(gl_FragCoord.x,gl_FragCoord.y);

vec2 ukeyframeTexCoord = vec2(ukeyframe.x*dx, ukeyframe.y*dy);
vec2 uframeTexCoord = vec2(uframe.x*dx, uframe.y*dy);

float f_pixel = texture(frame, uframeTexCoord).x*255.0;
float kf_pixel = texture(keyframe, ukeyframeTexCoord).x*255.0;

f_residual = pow((f_pixel-kf_pixel),2.0);
f_vertexID = vec3(g_vertexID[0], g_vertexID[1], g_vertexID[2]);
}

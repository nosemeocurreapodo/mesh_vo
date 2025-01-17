#pragma once

#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
#define MESH_WIDTH 32
#define MESH_HEIGHT 32
#define MAX_VERTEX_SIZE MESH_WIDTH*MESH_HEIGHT
#define MAX_TRIANGLE_SIZE (MESH_WIDTH-1)*(MESH_HEIGHT-1)*2
#define NUM_FRAMES 3
#define LAST_MIN_ANGLE M_PI/512.0
#define KEY_MAX_ANGLE M_PI/8.0
#define MIN_VIEW_PERC 0.8
#define MIN_LAMBDA 0.00001f
#define HUBER_THRESH_PIX 3.0
#define INITIAL_POSE_STD 0.01
#define INITIAL_PARAM_STD 0.1
#define RENDERER_NTHREADS 1
#define REDUCER_NTHREADS 1

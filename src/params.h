#pragma once

#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
#define MESH_WIDTH 32
#define MESH_HEIGHT 32
#define MAX_VERTEX_SIZE MESH_WIDTH *MESH_HEIGHT
#define MAX_TRIANGLE_SIZE (MESH_WIDTH - 1) * (MESH_HEIGHT - 1) * 2
#define NUM_FRAMES 5
#define LAST_MIN_ANGLE M_PI / 2048.0
#define KEY_MAX_ANGLE M_PI / 10.0
#define MIN_VIEW_PERC 0.7
#define MIN_LAMBDA 0.00001f
#define HUBER_THRESH_PIX 3.0
// for a max depth of 10, using log(depth) as param
#define MAX_PARAM fromDepthToParam(10.0)
#define MIN_PARAM fromDepthToParam(1.0)
#define INITIAL_POSE_STD 10.0
#define GOOD_POSE_STD INITIAL_POSE_STD * 0.5
// #define INITIAL_PARAM_STD fromDepthToParam(0.01)
// for a param = log(depth) and a max depth of 10, the max uncertanty should be ln(3) = 1.098
#define INITIAL_PARAM_STD fromDepthToParam(3.0)
// lets suppouse that by the optimization we reduce the uncertainty by 10x, that means a depth uncertainty of exp(0.1*1.098) = 1.116, or for 2x reduction, that means 1.731 uncertanty
#define GOOD_PARAM_STD INITIAL_PARAM_STD * 0.5
#define RENDERER_NTHREADS 1
#define REDUCER_NTHREADS 1

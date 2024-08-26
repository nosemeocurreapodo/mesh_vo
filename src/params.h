#pragma once

#define MAX_LEVELS 10 //max level for a 512x512 image
#define MESH_WIDTH 48
#define MESH_HEIGHT 48
#define NUM_FRAMES 3
#define HUBER_THRESH 30.0
#define RENDERER_NTHREADS 4
#define REDUCER_NTHREADS 4
//for depth
//#define INITIAL_VAR (10.0 * 10.0)
//for idepth = 1/depth
#define INITIAL_VAR (1.0/(10.0 * 10.0))
//for log(depth) = z
//#define INITIAL_VAR (std::log(10.0) * std::log(10.0))

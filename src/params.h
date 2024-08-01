#pragma once

#define MAX_LEVELS 10 //max level for a 512x512 image
#define MESH_WIDTH 24
#define MESH_HEIGHT 24
//for depth
//#define INITIAL_VAR (10.0 * 10.0)
//for idepth = 1/depth
#define INITIAL_VAR (1.0/(10.0 * 10.0))
//for log(depth) = z
//#define INITIAL_VAR (std::log(10.0) * std::log(10.0))

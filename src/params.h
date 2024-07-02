#pragma once

#define MAX_WIDTH 512
#define MAX_HEIGHT 512
#define MAX_LEVELS 9 //max level for a 512x512 image
#define NUM_FRAMES 1
#define MESH_WIDTH 16
#define MESH_HEIGHT 16
#define MIN_TRIANGLE_AREA (float(MAX_WIDTH)/MESH_WIDTH)*(float(MAX_HEIGHT)/MESH_HEIGHT)/32.0
#define MIN_TRIANGLE_ANGLE M_PI/32.0

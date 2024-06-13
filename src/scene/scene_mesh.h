#pragma once

#include <Eigen/Core>

#include "common/camera.h"
#include "cpu/frame_cpu.h"
#include "params.h"

class sceneMesh
{
public:
    sceneMesh()
    {
        // preallocate scene vertices to zero
        for (int y = 0; y < VERTEX_HEIGH; y++)
        {
            for (int x = 0; x < VERTEX_WIDTH; x++)
            {
                scene_vertices.push_back(0.0);
                scene_vertices.push_back(0.0);
                scene_vertices.push_back(0.0);

                /*
                if (x > 0 && y > 0)
                {
                    scene_indices.push_back(0);
                    scene_indices.push_back(0);
                    scene_indices.push_back(0);

                    scene_indices.push_back(0);
                    scene_indices.push_back(0);
                    scene_indices.push_back(0);
                }
                */
            }
        }

        // init scene indices
        for (int y = 0; y < VERTEX_HEIGH; y++)
        {
            for (int x = 0; x < VERTEX_WIDTH; x++)
            {
                if (x > 0 && y > 0)
                {
                    if (((x % 2 == 0)))
                    // if(((x % 2 == 0) && (y % 2 == 0)) || ((x % 2 != 0) && (y % 2 != 0)))
                    // if (rand() > 0.5 * RAND_MAX)
                    {
                        scene_indices.push_back(x - 1 + y * (VERTEX_WIDTH));
                        scene_indices.push_back(x + (y - 1) * (VERTEX_WIDTH));
                        scene_indices.push_back(x - 1 + (y - 1) * (VERTEX_WIDTH));

                        scene_indices.push_back(x + y * (VERTEX_WIDTH));
                        scene_indices.push_back(x + (y - 1) * (VERTEX_WIDTH));
                        scene_indices.push_back(x - 1 + y * (VERTEX_WIDTH));
                    }
                    else
                    {
                        scene_indices.push_back(x + y * (VERTEX_WIDTH));
                        scene_indices.push_back(x - 1 + (y - 1) * (VERTEX_WIDTH));
                        scene_indices.push_back(x - 1 + y * (VERTEX_WIDTH));

                        scene_indices.push_back(x + y * (VERTEX_WIDTH));
                        scene_indices.push_back(x + (y - 1) * (VERTEX_WIDTH));
                        scene_indices.push_back(x - 1 + (y - 1) * (VERTEX_WIDTH));
                    }
                }
            }
        }
    }

    void initWithRandomIdepth(camera &cam);
    void initWithIdepth(frameCpu &frame, camera &cam);

    // scene
    std::vector<float> scene_vertices;
    std::vector<unsigned int> scene_indices;

private:
};

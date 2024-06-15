#include "rayDepthMeshScene.h"

rayDepthMeshScene::rayDepthMeshScene(float fx, float fy, float cx, float cy, int width, int height)
    : cam(fx, fy, cx, cy, width, height)
{
    // preallocate scene vertices to zero
    for (int y = 0; y < VERTEX_HEIGH; y++)
    {
        for (int x = 0; x < VERTEX_WIDTH; x++)
        {
            scene_vertices.push_back(0.0);
            scene_vertices.push_back(0.0);
            scene_vertices.push_back(0.0);
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

void rayDepthMeshScene::init(frameCpu &f)
{
    f.copyTo(frame);
    setIdepth();
}

void rayDepthMeshScene::setIdepth()
{
    scene_vertices.clear();

    int lvl = 0;

    for (int y = 0; y < VERTEX_HEIGH; y++) // los vertices estan en el medio del pixel!! sino puede agarrar una distancia u otra
    {
        for (int x = 0; x < VERTEX_WIDTH; x++)
        {
            float xi = (float(x) / float(VERTEX_WIDTH - 1)) * cam.width[lvl];
            float yi = (float(y) / float(VERTEX_HEIGH - 1)) * cam.height[lvl];

            float idepth = frame.idepth.get(yi, xi, lvl);
            /*
            if(idepth <= min_idepth)
                idepth = min_idepth;
            if(idepth > max_idepth)
                idepth = max_idepth;
                */
            if (idepth != idepth || idepth < 0.1 || idepth > 1.0)
                idepth = 0.1 + (1.0 - 0.1) * float(y) / VERTEX_HEIGH;

            Eigen::Vector3f u = Eigen::Vector3f(xi, yi, 1.0);
            Eigen::Vector3f r = Eigen::Vector3f(cam.fxinv[lvl] * u(0) + cam.cxinv[lvl], cam.fyinv[lvl] * u(1) + cam.cyinv[lvl], 1.0);
            Eigen::Vector3f p = r / idepth;

            scene_vertices.push_back(r(0));
            scene_vertices.push_back(r(1));
            scene_vertices.push_back(idepth);
        }
    }
}

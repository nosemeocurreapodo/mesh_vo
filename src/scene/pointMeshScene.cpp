#include "pointMeshScene.h"

void pointMesh::initWithRandomIdepth(camera &cam)
{
    // const float* maxGradients = new_frame->maxGradients();

    scene_vertices.clear();

    for (int y = 0; y < VERTEX_HEIGH; y++) // los vertices estan en el medio del pixel!! sino puede agarrar una distancia u otra
    {
        for (int x = 0; x < VERTEX_WIDTH; x++)
        {
            float idepth = 0.1 + (1.0 - 0.1) * float(y) / VERTEX_HEIGH;
            // float idepth = 0.5f + 1.0f * ((rand() % 100001) / 100000.0f);
            // float idepth = 0.5;

            float xi = (float(x) / float(VERTEX_WIDTH - 1)) * cam.width[0];
            float yi = (float(y) / float(VERTEX_HEIGH - 1)) * cam.height[0];
            Eigen::Vector3f u = Eigen::Vector3f(xi, yi, 1.0);
            Eigen::Vector3f r = Eigen::Vector3f(cam.fxinv[0] * u(0) + cam.cxinv[0], cam.fyinv[0] * u(1) + cam.cyinv[0], 1.0);
            Eigen::Vector3f p = r / idepth;

            scene_vertices.push_back(r(0));
            scene_vertices.push_back(r(1));
            scene_vertices.push_back(idepth);
        }
    }
}

void pointMesh::initWithIdepth(data_cpu<float> &data_idepth, camera &cam, int lvl)
{
    scene_vertices.clear();

    for (int y = 0; y < VERTEX_HEIGH; y++) // los vertices estan en el medio del pixel!! sino puede agarrar una distancia u otra
    {
        for (int x = 0; x < VERTEX_WIDTH; x++)
        {
            float xi = (float(x) / float(VERTEX_WIDTH - 1)) * cam.width[lvl];
            float yi = (float(y) / float(VERTEX_HEIGH - 1)) * cam.height[lvl];

            float idepth = data_idepth.get(yi, xi, lvl);
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

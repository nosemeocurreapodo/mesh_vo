#include "scene_mesh.h"

void sceneMesh::initWithRandomIdepth(camera &cam)
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

void sceneMesh::initWithIdepth(frameCpu &frame, camera &cam)
{
    scene_vertices.clear();

    for (int y = 0; y < VERTEX_HEIGH; y++) // los vertices estan en el medio del pixel!! sino puede agarrar una distancia u otra
    {
        for (int x = 0; x < VERTEX_WIDTH; x++)
        {
            float xi = (float(x) / float(VERTEX_WIDTH - 1)) * cam.width[0];
            float yi = (float(y) / float(VERTEX_HEIGH - 1)) * cam.height[0];

            float idepth = frame.idepth.texture[0].at<float>(yi, xi);
            /*
            if(idepth <= min_idepth)
                idepth = min_idepth;
            if(idepth > max_idepth)
                idepth = max_idepth;
                */
            if (idepth != idepth || idepth < 0.1 || idepth > 1.0)
                idepth = 0.1 + (1.0 - 0.1) * float(y) / VERTEX_HEIGH;

            Eigen::Vector3f u = Eigen::Vector3f(xi, yi, 1.0);
            Eigen::Vector3f r = Eigen::Vector3f(cam.fxinv[0] * u(0) + cam.cxinv[0], cam.fyinv[0] * u(1) + cam.cyinv[0], 1.0);
            Eigen::Vector3f p = r / idepth;

            scene_vertices.push_back(r(0));
            scene_vertices.push_back(r(1));
            scene_vertices.push_back(idepth);
        }
    }
}

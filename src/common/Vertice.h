#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "common/camera.h"
#include "common/common.h"
//#include "common/Triangle.h"

class Triangle;

class Vertice
{
public:
    Vertice(Eigen::Vector3f &pos, Eigen::Vector2f &tc, unsigned int i)
    {
        position = pos;
        texcoord = tc;
        id = i;
    };

    //can be either rayidepth or a 3d position
    Eigen::Vector3f position;
    //normalized texcoords
    Eigen::Vector2f texcoord;
    unsigned int id;

private:
};

inline Vertice getClosestVertice(std::vector<Vertice> &v_vector, Vertice &v)
{
    float closest_distance = std::numeric_limits<float>::max();
    Vertice closest_vertice = v;
    for (int i = 0; i < (int)v_vector.size(); i++)
    {
        float distance = (v_vector[i].texcoord - v.texcoord).norm();
        
        //float x_distance = std::fabs(v_vector[i].texcoord(0) - v.texcoord(0));
        //float y_distance = std::fabs(v_vector[i].texcoord(1) - v.texcoord(1));
        //float distance = std::max(x_distance, y_distance);

        if (distance < closest_distance)
        {
            closest_distance = distance;
            closest_vertice = v_vector[i];
        }
    }
    return closest_vertice;
}
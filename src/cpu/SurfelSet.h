#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "cpu/PointSetNormals.h"
#include "cpu/Polygon.h"
#include "common/common.h"
#include "common/DelaunayTriangulation.h"
#include "params.h"

class SurfelSet : public PointSetNormals
{
public:
    SurfelSet() : PointSetNormals()
    {
    };

    SurfelSet(const SurfelSet &other) : PointSetNormals(other)
    {

    }

    SurfelSet &operator=(const SurfelSet &other)
    {
        if (this != &other)
        {
            PointSetNormals::operator=(other);
        }
        return *this;
    }

    void clear()
    {
        PointSet::clear();
    }

    PolygonCircle getPolygon(unsigned int id)
    {
        Eigen::Vector3f center = getVertice(id);
        Eigen::Vector3f normal = getNormal(id);
        PolygonCircle t(center, normal, radius);
        return t;
    }

private:
    float radius;
};

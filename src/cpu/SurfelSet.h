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

    PolygonFlat getPolygon(unsigned int id)
    {
        Eigen::Vector3f center = getVertice(id);
        Eigen::Vector3f normal = getNormal(id);
        PolygonFlat t(getVertice(tri[0]), getVertice(tri[1]), getVertice(tri[2]),
                      getTexCoord(tri[0]), getTexCoord(tri[1]), getTexCoord(tri[2]));
        return t;
    }

private:
    float radius;
};

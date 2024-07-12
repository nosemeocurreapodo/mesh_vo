#pragma once
#include <memory>
#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "cpu/PointSetNormals.h"
#include "cpu/Polygon.h"
#include "common/common.h"
#include "params.h"

class SurfelSet : public PointSetNormals
{
public:
    SurfelSet() : PointSetNormals()
    {
        radius = 10;
        setJackMethod(depthJacobian);
    };

    SurfelSet(const SurfelSet &other) : PointSetNormals(other)
    {
        radius = other.radius;
    }
    /*
    SurfelSet &operator=(const SurfelSet &other)
    {
        if (this != &other)
        {
            PointSetNormals::operator=(other);
        }
        return *this;
    }
    */
    std::unique_ptr<PointSet> clone() const override
    {
        return std::make_unique<SurfelSet>(*this);
    }

    void init(frameCPU &frame, camera &cam, dataCPU<float> &idepth, int lvl)
    {
        clear();

        for (float y = 0.0; y < MESH_HEIGHT; y++)
        {
            for (float x = 0.0; x < MESH_WIDTH; x++)
            {
                Eigen::Vector2f pix;
                pix[0] = (cam.width - 1) * x / (MESH_WIDTH - 1);
                pix[1] = (cam.height - 1) * y / (MESH_HEIGHT - 1);
                Eigen::Vector3f ray = cam.pixToRay(pix);
                float idph = idepth.get(pix[1], pix[0], lvl);
                if (idph == idepth.nodata)
                    continue;

                if (idph <= 0.0)
                    continue;

                Eigen::Vector3f vertice = ray / idph;

                unsigned int id = addVertice(vertice);
            }
        }

        setPose(frame.pose);
    }

    void clear()
    {
        PointSet::clear();
    }

    std::unique_ptr<Polygon> getPolygon(unsigned int id)
    {
        Eigen::Vector3f center = getVertice(id);
        Eigen::Vector3f normal = getNormal(id);
        PolygonCircle pol(center, normal, radius, getJacMethod());
        return std::make_unique<PolygonCircle>(pol);
    }

    std::vector<unsigned int> getPolygonsIds() const
    {
        return getVerticesIds();
    }

    /*
    std::vector<unsigned int> getPolygonVerticesIds(unsigned int id)
    {
        std::vector<unsigned int> ids;
        ids.push_back(id);
        return ids;
    }
    */

    std::vector<unsigned int> getPolygonParamsIds(unsigned int polId) override
    {
        // the id of the param (depth, nx, ny) is the id of the vertice * 3 + x
        std::vector<unsigned int> ids;
        ids.push_back(polId * 3);     // for the depth
        ids.push_back(polId * 3 + 1); // for nx
        ids.push_back(polId * 3 + 2); // for ny
        return ids;
    }

    void setParam(float param, unsigned int paramId) override
    {
        unsigned int verticeId = (unsigned int)paramId / 3;
        // if it is multiple of 3, its the depth
        if (paramId % 3 == 0)
        {
            if (param > 0.01 && param < 100.0)
                setVerticeDepth(param, verticeId);
        }
        // otherwise its the normal
        else
        {
            Eigen::Vector3f normal = getNormal(verticeId);
            if (paramId % 3 == 1)
                normal(0) = param;
            else
                normal(1) = param;
            Eigen::Vector3f vertice = getVertice(verticeId);
            Eigen::Vector3f ray = vertice/vertice(2);
            if(fabs(ray.dot(normal)) > 0.0000001)
                setNormal(normal, verticeId);
        }
    }

    float getParam(unsigned int paramId) override
    {
        unsigned int verticeId = (unsigned int)paramId / 3;
        // if it is multiple of 3, its the depth
        if (paramId % 3 == 0)
        {
            return getVerticeDepth(verticeId);
        }
        // otherwise its the normal
        else
        {
            Eigen::Vector3f normal = getNormal(verticeId);
            if (paramId % 3 == 1)
                return normal(0);
            else
                return normal(1);
        }
    }

private:
    float radius;
};

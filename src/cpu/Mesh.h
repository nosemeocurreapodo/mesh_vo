#pragma once
#include <memory>
#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "cpu/PointSet.h"
#include "cpu/Polygon.h"
#include "common/common.h"
#include "common/DelaunayTriangulation.h"
#include "params.h"

class Mesh : public PointSet
{
public:
    Mesh() : PointSet()
    {
        last_triangle_id = 0;
        setJackMethod(depthJacobian);
    };

    Mesh(const Mesh &other) : PointSet(other)
    {
        triangles = other.triangles;
        last_triangle_id = other.last_triangle_id;
    }
    /*
    Mesh &operator=(const Mesh &other)
    {
        if (this != &other)
        {
            PointSet::operator=(other);
            triangles = other.triangles;
            last_triangle_id = other.last_triangle_id;
        }
        return *this;
    }
    */

    std::unique_ptr<PointSet> clone() const override
    {
        return std::make_unique<Mesh>(*this);
    }

    void init(frameCPU &frame, camera &cam, dataCPU<float> &idepth, int lvl) override
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
        buildTriangles();
    }

    void clear()
    {
        PointSet::clear();
        triangles.clear();
        last_triangle_id = 0;
    }

    std::unique_ptr<Polygon> getPolygon(unsigned int polId)
    {
        // always return triangle in cartesian
        std::array<unsigned int, 3> tri = getTriangleIndices(polId);
        PolygonFlat pol(getVertice(tri[0]), getVertice(tri[1]), getVertice(tri[2]), getJacMethod());
        return std::make_unique<PolygonFlat>(pol);
    }

    std::vector<unsigned int> getPolygonsIds() const override
    {
        std::vector<unsigned int> keys;
        for (auto it = triangles.begin(); it != triangles.end(); ++it)
        {
            keys.push_back(it->first);
        }
        return keys;
    }

    /*
    std::vector<unsigned int> getPolygonVerticesIds(unsigned int id) override
    {
        std::vector<unsigned int> ids;
        std::array<unsigned int, 3> i = getTriangleIndices(id);
        ids.push_back(i[0]);
        ids.push_back(i[1]);
        ids.push_back(i[2]);
        return ids;
    }
    */

    std::vector<unsigned int> getPolygonParamsIds(unsigned int polId) override
    {
        // the id of the param (the depth in this case) is just the id of the vertice
        std::vector<unsigned int> ids;
        std::array<unsigned int, 3> i = getTriangleIndices(polId);
        ids.push_back(i[0]);
        ids.push_back(i[1]);
        ids.push_back(i[2]);
        return ids;
    }

    void setParam(float param, unsigned int paramId) override
    {
        if (getJacMethod() == MapJacobianMethod::depthJacobian)
            setVerticeDepth(param, paramId);
        if (getJacMethod() == MapJacobianMethod::idepthJacobian)
            setVerticeDepth(1.0/param, paramId);
        if (getJacMethod() == MapJacobianMethod::logDepthJacobian)
            setVerticeDepth(std::exp(param), paramId);
        if (getJacMethod() == MapJacobianMethod::logIdepthJacobian)
            setVerticeDepth(1.0/std::exp(param), paramId);

        // set the param (the depth in this case)
        //if (param > 0.01 && param < 100.0)
        //    setVerticeDepth(param, paramId);
    }

    float getParam(unsigned int paramId) override
    {
        if (getJacMethod() == MapJacobianMethod::depthJacobian)
            return getVerticeDepth(paramId);
        if (getJacMethod() == MapJacobianMethod::idepthJacobian)
            return 1.0/getVerticeDepth(paramId);
        if (getJacMethod() == MapJacobianMethod::logDepthJacobian)
            return std::log(getVerticeDepth(paramId));
        if (getJacMethod() == MapJacobianMethod::logIdepthJacobian)
            return std::log(1.0 / getVerticeDepth(paramId));

        // set the param (the depth in this case)
        return getVerticeDepth(paramId);
    }

private:
    void clearTriangles()
    {
        triangles.clear();
    }

    void setTriangles(std::map<unsigned int, std::array<unsigned int, 3>> &new_tris)
    {
        triangles = new_tris;
    }

    void removeTriangle(unsigned int id)
    {
        triangles.erase(id);
    }

    unsigned int addTriangle(std::array<unsigned int, 3> &tri)
    {
        last_triangle_id++;
        if (triangles.count(last_triangle_id))
            throw std::out_of_range("addTriangle id already exist");
        triangles[last_triangle_id] = tri;
        return last_triangle_id;
    }

    void setTriangleIndices(std::array<unsigned int, 3> &tri, unsigned int id)
    {
        if (!triangles.count(id))
            throw std::out_of_range("setTriangleIndices invalid id");
        triangles[id] = tri;
    }

    std::array<unsigned int, 3> getTriangleIndices(unsigned int id)
    {
        if (!triangles.count(id))
            throw std::out_of_range("setTriangleIndices invalid id");
        return triangles[id];
    }

    std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> computeEdgeFront()
    {
        std::vector<std::pair<std::array<unsigned int, 2>, unsigned int>> edgeFront;
        for (auto it = triangles.begin(); it != triangles.end(); ++it)
        {
            auto triIndices = it->second;

            // Triangle3D tri3D = getTriangle3D(it->first);
            // if (tri3D.isBackFace())
            //     continue;

            std::array<unsigned int, 2> edges[3];
            edges[0] = {triIndices[0], triIndices[1]};
            edges[1] = {triIndices[1], triIndices[2]};
            edges[2] = {triIndices[2], triIndices[0]};

            for (int i = 0; i < 3; i++)
            {
                int edge_index = -1;
                for (int j = 0; j < edgeFront.size(); j++)
                {
                    std::array<unsigned int, 2> ef = edgeFront[j].first;
                    unsigned int t_id = edgeFront[j].second;
                    if (isEdgeEqual(edges[i], ef))
                    {
                        edge_index = j;
                        break;
                    }
                }
                if (edge_index >= 0)
                    edgeFront.erase(edgeFront.begin() + edge_index);
                else
                    edgeFront.push_back({edges[i], it->first});
            }
        }
        return edgeFront;
    }

    bool isTrianglePresent(std::array<unsigned int, 3> &tri)
    {
        for (auto it = triangles.begin(); it != triangles.end(); ++it)
        {
            std::array<unsigned int, 3> tri2 = it->second;

            if (isTriangleEqual(tri, tri2))
                return true;
        }
        return false;
    }

    void removePointsWithoutTriangles()
    {
        std::vector<unsigned int> vertsIds = getVerticesIds();
        for (auto it = vertsIds.begin(); it != vertsIds.end(); it++)
        {
            bool remove = true;
            for (auto t_it = triangles.begin(); t_it != triangles.end(); t_it++)
            {
                if (*it == t_it->second[0] || *it == t_it->second[1] || *it == t_it->second[2])
                {
                    remove = false;
                    break;
                }
            }
            if (remove)
                removeVertice(*it);
        }
    }

    void removeTrianglesWithoutPoints()
    {
        std::vector<unsigned int> trisIds = getPolygonsIds();
        for (auto it = trisIds.begin(); it != trisIds.end(); it++)
        {
            std::array<unsigned int, 3> vertIds = getTriangleIndices(*it);
            try
            {
                PolygonFlat pol(getVertice(vertIds[0]), getVertice(vertIds[1]), getVertice(vertIds[2]), getJacMethod());
            }
            catch (std::string error)
            {
                triangles.erase(*it);
            }
        }
    }

    void buildTriangles()
    {
        DelaunayTriangulation triangulation;
        std::map<unsigned int, Eigen::Vector2f> rays;
        std::vector<unsigned int> ids = getVerticesIds();
        for (auto id : ids)
        {
            Eigen::Vector3f ray = getVertice(id) / getVertice(id)(2);
            rays[id](0) = ray(0);
            rays[id](1) = ray(1);
        }
        triangulation.loadPoints(rays);
        triangulation.triangulate();
        std::map<unsigned int, std::array<unsigned int, 3>> tris = triangulation.getTriangles();
        clearTriangles();
        setTriangles(tris);
    }
    /*
    void removeOcluded(camera &cam)
    {
        projectToCamera(cam);
        std::vector<unsigned int> trisIds = getTrianglesIds();
        for (auto it = trisIds.begin(); it != trisIds.end(); it++)
        {
            // remove triangle if:
            // 1-is backface
            // 2-the points are behind the camera
            // 2-all vertices lays outside the image
            // afterwards,
            // remove vertices without triangles
            Polygon pol = getPolygon(*it);
            if (pol.vertices[0](2) <= 0.0 || pol.vertices[1](2) <= 0.0 || pol.vertices[2](2) <= 0.0)
            {
                removeTriangle(*it);
                continue;
            }
            if (pol.getArea() < 1.0)
            {
                removeTriangle(*it);
                continue;
            }
            if (!cam.isPixVisible(pol.vertices[0]) && !cam.isPixVisible(pol.vertices[1]) && !cam.isPixVisible(pol.vertices[2]))
            {
                removeTriangle(*it);
                continue;
            }
        }
        removePointsWithoutTriangles();
    }
    */

    std::map<unsigned int, std::array<unsigned int, 3>> triangles;
    int last_triangle_id;
};

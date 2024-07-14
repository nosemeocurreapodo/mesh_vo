#pragma once
#include <memory>
#include <Eigen/Core>
#include "sophus/se3.hpp"
#include "cpu/PointSetNormals.h"
#include "cpu/Polygon.h"
#include "common/Error.h"
#include "common/HGMapped.h"
#include "common/common.h"
#include "common/DelaunayTriangulation.h"
#include "params.h"

class MeshSmooth : public PointSetNormals
{
public:
    MeshSmooth() : PointSetNormals()
    {
        last_triangle_id = 0;
        setJackMethod(idepthJacobian);
    };

    MeshSmooth(const MeshSmooth &other) : PointSetNormals(other)
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
        return std::make_unique<MeshSmooth>(*this);
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
                setNormal(ray, id);
            }
        }

        setPose(frame.pose);
        buildTriangles();
    }

    void clear()
    {
        PointSetNormals::clear();
        triangles.clear();
        last_triangle_id = 0;
    }

    std::unique_ptr<Polygon> getPolygon(unsigned int polId)
    {
        // always return triangle in cartesian
        std::array<unsigned int, 3> tri = getTriangleIndices(polId);
        PolygonSmooth pol(getVertice(tri[0]), getVertice(tri[1]), getVertice(tri[2]),
                          getNormal(tri[0]), getNormal(tri[1]), getNormal(tri[2]),
                          getJacMethod());
        return std::make_unique<PolygonSmooth>(pol);
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
        float new_depth;
        if (getJacMethod() == MapJacobianMethod::depthJacobian)
            new_depth = param;
        if (getJacMethod() == MapJacobianMethod::idepthJacobian)
            new_depth = 1.0 / param;
        if (getJacMethod() == MapJacobianMethod::logDepthJacobian)
            new_depth = std::exp(param);
        if (getJacMethod() == MapJacobianMethod::logIdepthJacobian)
            new_depth = 1.0 / std::exp(param);

        // set the param (the depth in this case)
        if (new_depth > 0.01 && new_depth < 100.0)
            setVerticeDepth(new_depth, paramId);
    }

    float getParam(unsigned int paramId) override
    {
        float param;
        if (getJacMethod() == MapJacobianMethod::depthJacobian)
            param = getVerticeDepth(paramId);
        if (getJacMethod() == MapJacobianMethod::idepthJacobian)
            param = 1.0 / getVerticeDepth(paramId);
        if (getJacMethod() == MapJacobianMethod::logDepthJacobian)
            param = std::log(getVerticeDepth(paramId));
        if (getJacMethod() == MapJacobianMethod::logIdepthJacobian)
            param = std::log(1.0 / getVerticeDepth(paramId));

        return param;
    }

    Error errorRegu()
    {
        Error error;

        std::vector<unsigned int> polIds = getPolygonsIds();

        for (size_t index = 0; index < polIds.size(); index++)
        {
            unsigned int id = polIds[index];
            std::array<unsigned int, 3> tri = getTriangleIndices(id);

            float theta[3];

            for (int j = 0; j < 3; j++)
            {
                if (getJacMethod() == MapJacobianMethod::depthJacobian)
                    theta[j] = getVerticeDepth(tri[j]);
                if (getJacMethod() == MapJacobianMethod::idepthJacobian)
                    theta[j] = 1.0 / getVerticeDepth(tri[j]);
                if (getJacMethod() == MapJacobianMethod::logDepthJacobian)
                    theta[j] = std::log(getVerticeDepth(tri[j]));
                if (getJacMethod() == MapJacobianMethod::logIdepthJacobian)
                    theta[j] = std::log(1.0 / getVerticeDepth(tri[j]));
            }

            if (theta[0] <= 0.0 || theta[1] <= 0.0 || theta[2] <= 0.0)
                continue;

            float diff1 = theta[0] - theta[1];
            float diff2 = theta[0] - theta[2];
            float diff3 = theta[1] - theta[2];

            error.error += diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
        }
        // divided by the number of triangles
        // we don't want to have less error if we have less triangles
        error.count = polIds.size();
        return error;
    }

    HGMapped HGRegu()
    {
        HGMapped hg;

        std::vector<unsigned int> polIds = getPolygonsIds();

        for (size_t i = 0; i < polIds.size(); i++)
        {
            unsigned int p_id = polIds[i];

            std::array<unsigned int, 3> v_ids = getTriangleIndices(p_id);

            float theta[v_ids.size()];

            for (int j = 0; j < v_ids.size(); j++)
            {
                if (getJacMethod() == MapJacobianMethod::depthJacobian)
                    theta[j] = getVerticeDepth(v_ids[j]);
                if (getJacMethod() == MapJacobianMethod::idepthJacobian)
                    theta[j] = 1.0 / getVerticeDepth(v_ids[j]);
                if (getJacMethod() == MapJacobianMethod::logDepthJacobian)
                    theta[j] = std::log(getVerticeDepth(v_ids[j]));
                if (getJacMethod() == MapJacobianMethod::logIdepthJacobian)
                    theta[j] = std::log(1.0 / getVerticeDepth(v_ids[j]));
            }

            if (theta[0] <= 0.0 || theta[1] <= 0.0 || theta[2] <= 0.0)
                continue;

            float diff1 = theta[0] - theta[1];
            float diff2 = theta[0] - theta[2];
            float diff3 = theta[1] - theta[2];

            float J1[3] = {1.0, -1.0, 0.0};
            float J2[3] = {1.0, 0.0, -1.0};
            float J3[3] = {0.0, 1.0, -1.0};

            for (int j = 0; j < 3; j++)
            {
                // if (hg.G(NUM_FRAMES*6 + vertexIndex[j]) == 0)
                //     continue;
                hg.G[v_ids[j]] += (diff1 * J1[j] + diff2 * J2[j] + diff3 * J3[j]);
                for (int k = 0; k < 3; k++)
                {
                    hg.H[v_ids[j]][v_ids[k]] += (J1[j] * J1[k] + J2[j] * J2[k] + J3[j] * J3[k]);
                }
            }
        }

        hg.count = polIds.size();

        return hg;
    }

    Error errorInitial(std::unique_ptr<PointSet> initScene, MatrixMapped &initThetaVar)
    {
        Error error;

        std::vector<unsigned int> vertsIds = getVerticesIds();

        for (size_t index = 0; index < vertsIds.size(); index++)
        {
            unsigned int id = vertsIds[index];

            float initVar = initThetaVar[id][id];
            float theta, initTheta;

            if (getJacMethod() == MapJacobianMethod::depthJacobian)
            {
                theta = getVerticeDepth(id);
                initTheta = initScene->getVerticeDepth(id);
            }
            if (getJacMethod() == MapJacobianMethod::idepthJacobian)
            {
                theta = 1.0 / getVerticeDepth(id);
                initTheta = 1.0 / initScene->getVerticeDepth(id);
            }
            if (getJacMethod() == MapJacobianMethod::logDepthJacobian)
            {
                theta = std::log(getVerticeDepth(id));
                initTheta = std::log(initScene->getVerticeDepth(id));
            }
            if (getJacMethod() == MapJacobianMethod::logIdepthJacobian)
            {
                theta = std::log(1.0 / getVerticeDepth(id));
                initTheta = std::log(1.0 / initScene->getVerticeDepth(id));
            }

            float diff = theta - initTheta;

            error.error += initVar * diff * diff;
        }
        // divided by the number of triangles
        // we don't want to have less error if we have less triangles
        error.count = vertsIds.size();
        return error;
    }

    HGMapped HGInitial(PointSet &initMesh, MatrixMapped &initThetaVar)
    {
        HGMapped hg;

        std::vector<unsigned int> vertsIds = getVerticesIds();

        for (size_t i = 0; i < vertsIds.size(); i++)
        {
            unsigned int v_id = vertsIds[i];

            float initVar = initThetaVar[v_id][v_id];
            float theta, initTheta;

            if (getJacMethod() == MapJacobianMethod::depthJacobian)
            {
                theta = getVerticeDepth(v_id);
                initTheta = initMesh.getVerticeDepth(v_id);
            }
            if (getJacMethod() == MapJacobianMethod::idepthJacobian)
            {
                theta = 1.0 / getVerticeDepth(v_id);
                initTheta = 1.0 / initMesh.getVerticeDepth(v_id);
            }
            if (getJacMethod() == MapJacobianMethod::logDepthJacobian)
            {
                theta = std::log(getVerticeDepth(v_id));
                initTheta = std::log(initMesh.getVerticeDepth(v_id));
            }
            if (getJacMethod() == MapJacobianMethod::logIdepthJacobian)
            {
                theta = std::log(1.0 / getVerticeDepth(v_id));
                initTheta = std::log(1.0 / initMesh.getVerticeDepth(v_id));
            }

            hg.G[v_id] += initVar * (theta - initTheta);
            hg.H[v_id][v_id] += initVar;
        }

        hg.count = vertsIds.size();

        return hg;
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

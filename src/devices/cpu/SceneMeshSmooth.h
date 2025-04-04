#pragma once

#include <memory>
#include "cpu/SceneVerticesNormalsBase.h"
#include "cpu/Shapes.h"
#include "common/Error.h"
#include "common/HGMapped.h"
#include "common/common.h"
#include "common/DelaunayTriangulation.h"
#include "params.h"

class SceneMeshSmooth : public SceneVerticesNormalsBase
{
public:
    SceneMeshSmooth() : SceneVerticesNormalsBase() {
                        };

    SceneMeshSmooth(const SceneMeshSmooth &other) : SceneVerticesNormalsBase(other)
    {
        triangles = other.triangles;
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

    std::unique_ptr<SceneBase> clone() const override
    {
        return std::make_unique<SceneMeshSmooth>(*this);
    }

    void clear()
    {
        SceneVerticesNormalsBase::clear();
        triangles.clear();
    }

    void init(frameCPU &frame, camera &cam, dataCPU<float> &idepth, dataCPU<float> &ivar, int lvl) override
    {
        clear();
        SceneVerticesBase::init(frame, cam, idepth, ivar, lvl);
        buildTriangles();
    }

    void init(frameCPU &frame, camera &cam, std::vector<vec3<float>> &vertices, int lvl) override
    {
        clear();
        SceneVerticesBase::init(frame, cam, vertices, lvl);
        buildTriangles();
    }

    void init(frameCPU &frame, camera &cam, std::vector<vec2<float>> &texcoords, std::vector<float> &idepths, int lvl) override
    {
        clear();
        SceneVerticesBase::init(frame, cam, texcoords, idepths, lvl);
        buildTriangles();
    }

    int getShapesDoF() override
    {
        return 3;
    }

    int getNumParams() override
    {
        // one depth for each vertice
        return getVerticesIds().size();
    }

    bool isShapeInWindow(window &win, int polId) override
    {
        auto tri = getTriangleIndices(polId);
        if (win.isPixInWindow(getPix(tri(0))) || win.isPixInWindow(getPix(tri(1))) || win.isPixInWindow(getPix(tri(2))))
            return true;
        return false;
    }

    std::unique_ptr<ShapeBase> getShape(camera cam, int polId) override
    {
        auto tri = getTriangleIndices(polId);
        // return std::make_unique<ShapeTriangleFlat>(getVertice(tri[0]), getVertice(tri[1]), getVertice(tri[2]), getDepthJacMethod());
        return std::make_unique<ShapeTriangleSmooth>(getRay(tri(0)), getRay(tri(1)), getRay(tri(2)),
                                                     getPix(tri(0)), getPix(tri(1)), getPix(tri(2)),
                                                     getDepth(tri(0)), getDepth(tri(1)), getDepth(tri(2)),
                                                     getWeight(tri(0)), getWeight(tri(1)), getWeight(tri(2)));
        // return std::make_unique<ShapeTriangleFlat>(getRay(tri[0]), getRay(tri[1]), getRay(tri[2]),
        //                                            getPix(tri[0]), getPix(tri[1]), getPix(tri[2]),
        //                                            getDepth(tri[0]), getDepth(tri[1]), getDepth(tri[2]),
        //                                            getDepthJacMethod());
    }

    void getShape(ShapeBase *shape, camera cam, int polId) override
    {
        auto tri = getTriangleIndices(polId);
        // return std::make_unique<ShapeTriangleFlat>(getVertice(tri[0]), getVertice(tri[1]), getVertice(tri[2]), getDepthJacMethod());
        ShapeTriangleSmooth *_shape = (ShapeTriangleSmooth *)shape;
        _shape->set(getRay(tri(0)), getRay(tri(1)), getRay(tri(2)),
                    getPix(tri(0)), getPix(tri(1)), getPix(tri(2)),
                    getDepth(tri(0)), getDepth(tri(1)), getDepth(tri(2)),
                    getWeight(tri(0)), getWeight(tri(1)), getWeight(tri(2)));
    }

    std::vector<int> getShapesIds() const override
    {
        return getTrianglesIds();
    }

    std::vector<int> getShapeParamsIds(int polId) override
    {
        // the id of the param (the depth in this case) is just the id of the vertice
        std::vector<int> ids;
        vec3<int> i = getTriangleIndices(polId);
        ids.push_back(i(0));
        ids.push_back(i(1));
        ids.push_back(i(2));
        return ids;
    }

    void setParam(float param, int paramId) override
    {
        setDepthParam(param, paramId);
    }

    void setParamWeight(float weight, int paramId) override
    {
        setWeight(weight, paramId);
    }

    float getParam(int paramId) override
    {
        return getDepthParam(paramId);
    }

    Error errorRegu()
    {
        Error error;

        std::vector<unsigned int> polIds = getTrianglesIds();

        for (size_t index = 0; index < polIds.size(); index++)
        {
            unsigned int id = polIds[index];
            vec3<int> tri = getTriangleIndices(id);

            float theta[3];

            for (int j = 0; j < 3; j++)
            {
                theta[j] = getDepthParam(tri[j]);
            }

            float diff1 = theta[0] - theta[1];
            float diff2 = theta[0] - theta[2];
            float diff3 = theta[1] - theta[2];

            error += diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
        }

        return error;
    }

    HGEigenSparse HGRegu(camera cam, int numFrames = 0)
    {
        std::vector<int> polIds = getTrianglesIds();

        HGEigenSparse hg(getNumParams() + numFrames * 8);

        for (size_t i = 0; i < polIds.size(); i++)
        {
            int p_id = polIds[i];

            vec3<int> v_ids = getTriangleIndices(p_id);

            float theta[3];
            theta[0] = getDepthParam(v_ids(0));
            theta[1] = getDepthParam(v_ids(1));
            theta[2] = getDepthParam(v_ids(2));

            float diff1 = theta[0] - theta[1];
            float diff2 = theta[0] - theta[2];
            float diff3 = theta[1] - theta[0];
            float diff4 = theta[1] - theta[2];
            float diff5 = theta[2] - theta[0];
            float diff6 = theta[2] - theta[1];

            vec3<float> J1(1.0, -1.0, 0.0);
            vec3<float> J2(1.0, 0.0, -1.0);
            vec3<float> J3(-1.0, 1.0, 0.0);
            vec3<float> J4(0.0, 1.0, -1.0);
            vec3<float> J5(-1.0, 0.0, 1.0);
            vec3<float> J6(0.0, -1.0, 1.0);

            hg.sparseAdd(J1, diff1, 1.0, v_ids);
            hg.sparseAdd(J2, diff2, 1.0, v_ids);
            // hg.sparseAdd(J3, diff3, 1.0, v_ids);
            hg.sparseAdd(J4, diff4, 1.0, v_ids);
            // hg.sparseAdd(J5, diff5, 1.0, v_ids);
            // hg.sparseAdd(J6, diff6, 1.0, v_ids);
        }

        hg.endSparseAdd();

        return hg;
    }

    Error errorInitial(SceneMesh &initScene, MatrixMapped &initThetaVar)
    {
        Error error;

        std::vector<int> vertsIds = getVerticesIds();

        for (size_t index = 0; index < vertsIds.size(); index++)
        {
            int id = vertsIds[index];

            float initVar = initThetaVar[id][id];

            float theta = getDepthParam(id);
            float initTheta = initScene.getDepthParam(id);

            float diff = theta - initTheta;

            error += initVar * diff * diff;
        }

        return error;
    }

    HGMapped HGInitial(SceneMesh &initMesh, MatrixMapped &initThetaVar)
    {
        HGMapped hg;

        /*
        std::vector<unsigned int> vertsIds = getVerticesIds();

        for (size_t i = 0; i < vertsIds.size(); i++)
        {
            unsigned int v_id = vertsIds[i];

            float initVar = initThetaVar[v_id][v_id];

            float theta = getDepthParam(v_id);
            float initTheta = initMesh.getDepthParam(v_id);

            hg.G[v_id] += initVar * (theta - initTheta);
            hg.H[v_id][v_id] += initVar;
        }

        hg.count = vertsIds.size();
        */

        return hg;
    }

private:
    void clearTriangles()
    {
        triangles.clear();
    }

    void setTriangles(std::vector<vec3<int>> new_tris)
    {
        triangles = new_tris;
    }

    void removeTriangle(unsigned int id)
    {
        triangles.erase(triangles.begin() + id);
    }

    unsigned int addTriangle(vec3<int> &tri)
    {
        int id = triangles.size();
        triangles.push_back(tri);
        return id;
    }

    void setTriangleIndices(vec3<int> &tri, int id)
    {
#ifdef DEBUG
        if (id >= triangles.size())
            throw std::out_of_range("setTriangleIndices invalid id");
#endif
        triangles[id] = tri;
    }

    inline vec3<int> &getTriangleIndices(int id)
    {
#ifdef DEBUG
        if (id >= triangles.size())
            throw std::out_of_range("setTriangleIndices invalid id");
#endif
        return triangles[id];
    }

    std::vector<int> getTrianglesIds() const
    {
        std::vector<int> keys;
        for (int it = 0; it < triangles.size(); ++it)
        {
            keys.push_back(it);
        }
        return keys;
    }
    /*
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
                for (size_t j = 0; j < edgeFront.size(); j++)
                {
                    std::array<unsigned int, 2> ef = edgeFront[j].first;
                    // unsigned int t_id = edgeFront[j].second;
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
    */

    /*
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
    */

    /*
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
             //if (remove)
             //    removeVertice(*it);
         }
     }
     */

    void removeTrianglesWithoutPoints()
    {
        /*
        std::vector<unsigned int> trisIds = getTrianglesIds();
        for (auto it = trisIds.begin(); it != trisIds.end(); it++)
        {
            std::array<unsigned int, 3> vertIds = getTriangleIndices(*it);
            try
            {
                ShapeTriangleFlat pol(getVertice(vertIds[0]), getVertice(vertIds[1]), getVertice(vertIds[2]), getDepthJacMethod());
            }
            catch (std::string error)
            {
                triangles.erase(*it);
            }
        }
        */
    }

    void buildTriangles()
    {
        DelaunayTriangulation triangulation;
        triangulation.loadPoints(getPixels());
        triangulation.triangulate();
        // clearTriangles();
        setTriangles(triangulation.getTriangles());
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

    std::vector<vec3<int>> triangles;
};

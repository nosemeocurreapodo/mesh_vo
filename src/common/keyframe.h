#pragma once

#include "params.h"
#include "common/types.h"
#include "core/mesh_helpers.h"

// #include <vector>
// #include <stdexcept>
// #include <algorithm>
// #include <cmath>

// 2D cross product (b-a) x (p-a)
template <typename T>
static T cross2(const Vec2<T> &a, const Vec2<T> &b, const Vec2<T> &p)
{
    return (b(0) - a(0)) * (p(1) - a(1)) - (b(1) - a(1)) * (p(0) - a(0));
}

// Point in triangle test in 2D, inclusive of edges.
// Works fine for pixel-space coordinates too.
template <typename T>
static bool pointInTri2D(const Vec2<T> &p, const Vec2<T> &a, const Vec2<T> &b, const Vec2<T> &c, T eps = (T)1e-6)
{
    const T c1 = cross2(a, b, p);
    const T c2 = cross2(b, c, p);
    const T c3 = cross2(c, a, p);

    const bool hasNeg = (c1 < -eps) || (c2 < -eps) || (c3 < -eps);
    const bool hasPos = (c1 > eps) || (c2 > eps) || (c3 > eps);

    // inside if not both negative and positive
    return !(hasNeg && hasPos);
}

static bool pointInMesh2D(
    const Vec2<float> &p,
    const std::vector<Vec2<float>> &texcoords, // [u0,v0,u1,v1,...] (pixels in your case)
    const std::vector<Vec3<int>> &indices      // [i0,i1,i2, i3,i4,i5, ...]
)
{
    const int triCount = (int)indices.size();

    for (int t = 0; t < triCount; ++t)
    {
        const Vec3<int> ind = indices[t];

        const Vec2<float> a = texcoords[ind(0)];
        const Vec2<float> b = texcoords[ind(1)];
        const Vec2<float> c = texcoords[ind(2)];

        // quick reject via AABB (cheap speed-up)
        const float minx = std::min({a(0), b(0), c(0)});
        const float maxx = std::max({a(0), b(0), c(0)});
        const float miny = std::min({a(1), b(1), c(1)});
        const float maxy = std::max({a(1), b(1), c(1)});

        // if (p(0) < minx || p(0) > maxx || p(1) < miny || p(1) > maxy)
        //     continue;

        Vec2<float> diff_0 = p - a;
        Vec2<float> diff_1 = p - b;
        Vec2<float> diff_2 = p - c;
        float dist_0 = sqrt(diff_0.dot(diff_0));
        float dist_1 = sqrt(diff_1.dot(diff_1));
        float dist_2 = sqrt(diff_2.dot(diff_2));

        if (dist_0 < 0.1 || dist_1 < 0.1 || dist_2 < 0.1)
            return true;

        if (pointInTri2D(p, a, b, c))
            return true;
    }
    return false;
}

class KeyFrame
{
public:
    /*
        keyFrame()
        {
            globalPose = SE3();
            globalExp = Vec2(0.0f, 0.0f);
            globalScale = 1.0;
        };
    */
    KeyFrame(const Texture<ImageType> &image, const Texture<Vec3f> &didxy, SE3f global_pose, const Mesh &mesh, float global_scale, int id)
        : image_(image), didxy_(didxy), global_pose_(global_pose), mesh_(mesh), global_scale_(global_scale), id_(id)
    {
    }

    KeyFrame(const KeyFrame &other)
        : image_(other.image_),
          didxy_(other.didxy_),
          global_pose_(other.global_pose_),
          mesh_(other.mesh_),
          global_scale_(other.global_scale_),
          id_(other.id_)
    {
    }

    KeyFrame &operator=(const KeyFrame &other)
    {
        if (this != &other)
        {
            image_ = other.image_;
            didxy_ = other.didxy_;
            global_pose_ = other.global_pose_;
            mesh_ = other.mesh_;
            global_scale_ = other.global_scale_;
            id_ = other.id_;
        }
        return *this;
    }

    int id()
    {
        return id_;
    }

    const Texture<ImageType> &image()
    {
        return image_;
    }

    const Texture<Vec3f> &didxy()
    {
        return didxy_;
    }

    const SE3f &global_pose()
    {
        return global_pose_;
    }

    const Mesh &mesh() const
    {
        return mesh_;
    }

    Mesh &mesh()
    {
        return mesh_;
    }

    float getGlobalScale()
    {
        return global_scale_;
    }

    SE3f localPoseToGlobal(SE3f localPose)
    {
        SE3f localPoseScaled = localPose;
        localPoseScaled.translation() *= global_scale_;
        SE3f globalPose = localPoseScaled * global_pose_;
        return globalPose;
    }

    SE3f globalPoseToLocal(SE3f globalPose)
    {
        SE3f localPose = globalPose * global_pose_.inverse();
        localPose.translation() /= global_scale_;
        return localPose;
    }
    /*
    Vec2f localExpToGlobal(const Vec2f &local_exposure)
    {
        Vec2f globalExp;
        float alpha1 = std::exp(local_exposure(0));
        float alpha2 = std::exp(global_exposure_(0));
        globalExp(0) = std::log(alpha1 / alpha2);
        globalExp(1) = local_exposure(1) - global_exposure_(1) / alpha1;
        return globalExp;
    }

    Vec2f globalExpToLocal(const Vec2f &global_exposure)
    {
        Vec2f localExp;
        float alpha1 = std::exp(-global_exposure(0));
        float alpha2 = std::exp(-global_exposure_(0));
        localExp(0) = std::log(alpha1 / alpha2);
        localExp(1) = -global_exposure(1) + (alpha2 / alpha1) * global_exposure_(1);
        return localExp;
    }
    */

    float meanDepth()
    {
        std::vector<Vec3<float>> vertices = get_vertices(mesh_);
        int size = vertices.size();
        float mean = 0.0;
        for (int i = 0; i < size; i++)
            mean += vertices[i](2);
        return mean / size;
    }

    void scaleMesh(float scale)
    {
        global_scale_ *= scale;
        std::vector<Vec3<float>> vertices = get_vertices(mesh_);
        for (int i = 0; i < vertices.size(); i++)
            vertices[i] = vertices[i] / scale;
        set_vertices(mesh_, vertices);
    }

    void changeFrame(const Texture<ImageType> &new_image,
                     const Texture<Vec3f> &new_didxy,
                     const SE3f &new_local_pose,
                     int new_id,
                     const Camera &cam)
    {
        std::vector<Vec3<float>> vertices = get_vertices(mesh_);
        std::vector<Vec2<float>> texcoords = get_texcoords(mesh_);
        std::vector<Vec3<int>> indices = get_indices(mesh_);

        for (int i = 0; i < vertices.size(); i++)
        {
            Vec3<float> vertex = vertices[i];

            vertex = new_local_pose * vertex;
            Vec3<float> ray = vertex / vertex(2);
            Vec2<float> pix = cam.RayToPix(ray);

            vertices[i] = vertex;
            texcoords[i] = pix;
        }

        std::vector<Vec2<float>> grid_uv; //= UniformTexCoords(mesh_vo::mesh_width, mesh_vo::mesh_height, 0.0, 0.0, 1.0, 1.0);

        std::vector<Vec3<float>> new_vertices;
        std::vector<Vec2<float>> new_texcoords;
        new_texcoords.reserve(grid_uv.size());

        for (int i = 0; i < (int)grid_uv.size(); ++i)
        {
            const Vec2<float> p = grid_uv[i];
            if (!pointInMesh2D(p, texcoords, indices))
            {
                Vec3<float> new_vertice = cam.PixToRay(p);
                new_texcoords.push_back(p);
                new_vertices.push_back(new_vertice);
            }
        }

        DelaunayTriangulation triangulator;
        triangulator.LoadMesh(texcoords, indices);
        triangulator.AddOutsidePoints(new_texcoords);
        // triangulator.LoadPoints(texcoords);
        // triangulator.Triangulate();
        std::vector<Vec3<int>> tris = triangulator.GetTriangles();

        std::vector<float> new_mesh_vertex;
        std::vector<int> new_indices;

        for (int i = 0; i < vertices.size(); i++)
        {
            Vec3<float> vertex = vertices[i];
            Vec2<float> texcoord = texcoords[i];
            new_mesh_vertex.push_back(vertex(0));
            new_mesh_vertex.push_back(vertex(1));
            new_mesh_vertex.push_back(vertex(2));
            new_mesh_vertex.push_back(texcoord(0));
            new_mesh_vertex.push_back(texcoord(1));
        }

        for (int i = 0; i < new_vertices.size(); i++)
        {
            Vec3<float> vertex = new_vertices[i];
            Vec2<float> texcoord = new_texcoords[i];
            new_mesh_vertex.push_back(vertex(0));
            new_mesh_vertex.push_back(vertex(1));
            new_mesh_vertex.push_back(vertex(2));
            new_mesh_vertex.push_back(texcoord(0));
            new_mesh_vertex.push_back(texcoord(1));
        }

        for (int i = 0; i < tris.size(); i++)
        {
            Vec3<int> tri = tris[i];
            new_indices.push_back(tri(0));
            new_indices.push_back(tri(1));
            new_indices.push_back(tri(2));
        }

        Mesh new_mesh(new_mesh_vertex, new_indices, true, true, false);

        image_ = new_image;
        didxy_ = new_didxy;
        global_pose_ = localPoseToGlobal(new_local_pose);
        mesh_ = new_mesh;
        id_ = new_id;
    }

    void changeFrame(const Texture<ImageType> &new_image,
                     const Texture<Vec3f> &new_didxy,
                     const Texture<float> &new_depth,
                     const SE3f &new_local_pose,
                     int new_id,
                     const Camera &cam)
    {
        std::vector<Vec2<float>> grid_uv = UniformTexCoords(mesh_vo::mesh_width, mesh_vo::mesh_height, 0.0, 0.0, 1.0, 1.0);

        std::vector<Vec3<float>> new_vertices;
        std::vector<Vec2<float>> new_texcoords;
        new_texcoords.reserve(grid_uv.size());

        auto depth_map = new_depth.MapRead(0);
        int width = new_depth.width(0);
        int height = new_depth.height(0);
        for (int i = 0; i < (int)grid_uv.size(); ++i)
        {
            const Vec2<float> p = grid_uv[i];
            Vec3<float> new_ray = cam.PixToRay(p);
            float depth = depth_map[p(1) * (height - 1) * width + p(0) * (width - 1)];
            Vec3<float> new_vertice = new_ray;
            if (depth != new_depth.nodata())
                new_vertice = new_ray * depth;
            new_texcoords.push_back(p);
            new_vertices.push_back(new_vertice);
        }

        DelaunayTriangulation triangulator;
        triangulator.LoadPoints(new_texcoords);
        triangulator.Triangulate();
        std::vector<Vec3<int>> tris = triangulator.GetTriangles();

        std::vector<float> new_mesh_vertex;
        std::vector<int> new_indices;

        for (int i = 0; i < new_vertices.size(); i++)
        {
            Vec3<float> vertex = new_vertices[i];
            Vec2<float> texcoord = new_texcoords[i];
            new_mesh_vertex.push_back(vertex(0));
            new_mesh_vertex.push_back(vertex(1));
            new_mesh_vertex.push_back(vertex(2));
            new_mesh_vertex.push_back(texcoord(0));
            new_mesh_vertex.push_back(texcoord(1));
        }

        for (int i = 0; i < tris.size(); i++)
        {
            Vec3<int> tri = tris[i];
            new_indices.push_back(tri(0));
            new_indices.push_back(tri(1));
            new_indices.push_back(tri(2));
        }

        Mesh new_mesh(new_mesh_vertex, new_indices, true, true, false);

        image_ = new_image;
        didxy_ = new_didxy;
        global_pose_ = localPoseToGlobal(new_local_pose);
        mesh_ = new_mesh;
        id_ = new_id;
    }

    float meanViewAngle(const SE3f &pose1, const SE3f &pose2)
    {
        int lvl = 1;

        // geometryType scene1 = geometry;
        // scene1.transform(pose1);
        //  scene1.project(cam);

        // geometryType scene2 = geometry;
        // scene2.transform(pose2);
        //  scene2.project(cam);

        std::vector<Vec3<float>> positions = get_vertices(mesh_);

        SE3f relativePose = pose1 * pose2.inverse();

        SE3f frame1PoseInv = relativePose.inverse();
        SE3f frame2PoseInv = SE3f();

        Vec3f frame1Translation = frame1PoseInv.translation();
        Vec3f frame2Translation = frame2PoseInv.translation();

        // std::vector<int> vIds = scene2.getVerticesIds();

        float accAngle = 0;
        int count = 0;
        for (int i = 0; i < positions.size(); i++)
        {
            Vec3f vert_ini = positions[i];
            Vec3f vert = pose2 * vert_ini;

            Vec3f diff1 = vert - frame1Translation;
            Vec3f diff2 = vert - frame2Translation;

            assert(diff1.norm() > 0 && diff2.norm() > 0);

            Vec3f diff1Normalized = diff1 / diff1.norm();
            Vec3f diff2Normalized = diff2 / diff2.norm();

            float cos_angle = diff1Normalized.dot(diff2Normalized);
            cos_angle = std::clamp(cos_angle, -1.0f, 1.0f);
            float angle = std::acos(cos_angle);

            assert(!std::isnan(angle));

            accAngle += angle;
            count += 1;
        }

        return accAngle / count;
    }

private:
    // Frame frame_;
    Texture<ImageType> image_;
    Texture<Vec3f> didxy_;
    Mesh mesh_;
    SE3f global_pose_;
    float global_scale_;
    int id_;
};

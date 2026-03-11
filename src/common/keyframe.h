#pragma once

#include "params.h"
#include "common/types.h"
#include "mpdr/common/mesh_helpers.h"

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
    // KeyFrame(const Texture<ImageType> &image, const Texture<Vec3f> didxy, Mesh mesh, SE3f global_pose, float global_scale, int id)
    //     : image_(image), didxy_(didxy), mesh_(std::move(mesh)), global_pose_(global_pose), global_scale_(global_scale), id_(id)
    //{
    // }
    KeyFrame(int width, int height, Mesh mesh)
        : image_(width, height, -1),
          mesh_(std::move(mesh)),
          global_scale_(1.0)
    {
    }

    KeyFrame(const KeyFrame &) = delete;
    KeyFrame &operator=(const KeyFrame &) = delete;

    KeyFrame(KeyFrame &&) noexcept = default;
    KeyFrame &operator=(KeyFrame &&) noexcept = default;

    // const Frame* frame() const
    //{
    //     return frame_;
    // }

    const Texture<ImageType> &image() const { return image_; }
    Texture<ImageType> &image() { return image_; }

    const int &id() const { return id_; }
    int &id() { return id_; }

    const Mesh &mesh() const { return mesh_; }
    Mesh &mesh() { return mesh_; }

    const SE3d &global_pose() const { return global_pose_; }
    SE3d &global_pose() { return global_pose_; }

    const double &global_scale() const { return global_scale_; }
    double &global_scale() { return global_scale_; }

    SE3d local_pose_to_global(SE3f local_pose) const
    {
        SE3d localPoseScaled; // = local_pose;
        Quaternion<float> fq = local_pose.so3().unit_quaternion();
        Quaternion<double> dq(fq.w(), fq.x(), fq.y(), fq.z());
        localPoseScaled.so3().setQuaternion(dq);
        Vec3<float> ft = local_pose.translation();
        Vec3<double> dt(ft(0), ft(1), ft(2));
        localPoseScaled.translation() = dt;
        localPoseScaled.translation() *= global_scale_;
        SE3d globalPose = localPoseScaled * global_pose_;
        return globalPose;
    }

    SE3f global_pose_to_local(SE3d global_pose) const
    {
        SE3d localPose = global_pose * global_pose_.inverse();
        localPose.translation() /= global_scale_;

        SE3f localPosef;
        Quaternion<double> dq = localPose.so3().unit_quaternion();
        Quaternion<float> fq(dq.w(), dq.x(), dq.y(), dq.z());
        Vec3<double> dt = localPose.translation();
        Vec3<float> ft(dt(0), dt(1), dt(2));
        localPosef.setQuaternion(fq);
        localPosef.translation() = ft;

        return localPosef;
    }

    void scale_mesh(double scale)
    {
        global_scale_ *= scale;
        std::vector<Vec3f> vertices = get_vertices(mesh_);
        for (int i = 0; i < vertices.size(); i++)
            vertices[i] = vertices[i] / float(scale);
        set_vertices(mesh_, vertices);
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

    float meanViewAngle(const SE3d &pose1, const SE3d &pose2, const Cameraf &cam) const
    {
        const std::vector<Vec3f> positions = get_vertices(mesh_);

        // Convert both poses into this keyframe's local frame
        const SE3f localpose1 = global_pose_to_local(pose1);
        const SE3f localpose2 = global_pose_to_local(pose2);

        // Transform from camera-1 frame to camera-2 frame:
        // x2 = T_2_1 * x1
        const SE3f T_2_1 = localpose2 * localpose1.inverse();

        // Camera-2 center expressed in camera-1 coordinates
        const Vec3f cam2_in_cam1 = T_2_1.inverse().translation();

        float accAngle = 0.0f;
        int count = 0;
        constexpr float eps = 1e-8f;

        for (size_t i = 0; i < positions.size(); ++i)
        {
            const Vec3f &vert = positions[i];

            // Vertex coordinates in each camera frame
            const Vec3f vert_cam1 = localpose1 * vert;
            const Vec3f vert_cam2 = localpose2 * vert;

            // Must be in front of both cameras
            if (vert_cam1(2) <= eps || vert_cam2(2) <= eps)
                continue;

            const Vec2f pix1 = cam.pointToPix(vert_cam1);
            const Vec2f pix2 = cam.pointToPix(vert_cam2);

            if (!cam.IsPixVisible(pix1) || !cam.IsPixVisible(pix2))
                continue;

            // Rays from each camera center to the same 3D point, expressed in camera-1 frame
            const Vec3f ray1 = vert_cam1;                // cam1 center is origin in cam1 frame
            const Vec3f ray2 = vert_cam1 - cam2_in_cam1; // cam2 center expressed in cam1 frame

            const float n1 = ray1.norm();
            const float n2 = ray2.norm();

            if (n1 <= eps || n2 <= eps)
                continue;

            float cos_angle = ray1.dot(ray2) / (n1 * n2);
            cos_angle = std::clamp(cos_angle, -1.0f, 1.0f);

            const float angle = std::acos(cos_angle);

            if (!std::isnan(angle))
            {
                accAngle += angle;
                ++count;
            }
        }

        return (count > 0) ? (accAngle / static_cast<float>(count)) : 0.0f;
    }

private:
    // Frame frame_;
    // const Frame *frame_;
    Texture<ImageType> image_;
    Mesh mesh_;
    SE3d global_pose_;
    double global_scale_;
    int id_;
};

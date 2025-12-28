#pragma once

#include "params.h"
#include "common/types.h"

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
    KeyFrame(const Frame &frame, Mesh &mesh, float global_scale)
        : frame_(frame), mesh_(mesh)
    {
        // global_exposure_ = global_exposure;
        global_scale_ = global_scale;
    }

    KeyFrame(const KeyFrame &other)
        : frame_(other.frame_),
          mesh_(other.mesh_),
          // global_exposure_(other.global_exposure_),
          global_scale_(other.global_scale_)
    {
    }

    KeyFrame &operator=(const KeyFrame &other)
    {
        if (this != &other)
        {
            frame_ = other.frame_;
            mesh_ = other.mesh_;
            global_exposure_ = other.global_exposure_;
            global_scale_ = other.global_scale_;
        }
        return *this;
    }

    const Frame &frame()
    {
        return frame_;
    }

    const Mesh &mesh() const
    {
        return mesh_;
    }

    Mesh &mesh()
    {
        return mesh_;
    }

    SE3f localPoseToGlobal(SE3f localPose)
    {
        SE3f localPoseScaled = localPose;
        localPoseScaled.translation() *= global_scale_;
        SE3f globalPose = localPoseScaled * frame_.global_pose();
        return globalPose;
    }

    SE3f globalPoseToLocal(SE3f globalPose)
    {
        SE3f localPose = globalPose * frame_.global_pose().inverse();
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

    float getGlobalScale()
    {
        return global_scale_;
    }

    float meanDepth()
    {
        std::vector<float> positions = mesh_.get_positions();
        int size = positions.size() / 3;
        float mean = 0.0;
        for (int i = 0; i < size; i++)
            mean += positions[i * 3 + 2];
        return mean / size;
    }

    void scaleMesh(float scale)
    {
        global_scale_ *= scale;
        std::vector<float> positions = mesh_.get_positions();
        for (int i = 0; i < positions.size(); i++)
            positions[i] = positions[i] / scale;
        mesh_.set_positions(positions);
    }

    void transformMesh(const SE3f &transformation)
    {
        std::vector<float> positions = mesh_.get_positions();
        for (int i = 0; i < positions.size(); i+=3)
        {
            Vec3<float> vert(positions[i], positions[i+1], positions[i+2]);
            vert = transformation * vert;
            positions[i] = vert(0);
            positions[i+1] = vert(1);
            positions[i+2] = vert(2);
        }
        mesh_.set_positions(positions);
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

        std::vector<float> positions = mesh_.get_positions();

        SE3f relativePose = pose1 * pose2.inverse();

        SE3f frame1PoseInv = relativePose.inverse();
        SE3f frame2PoseInv = SE3f();

        Vec3f frame1Translation = frame1PoseInv.translation();
        Vec3f frame2Translation = frame2PoseInv.translation();

        // std::vector<int> vIds = scene2.getVerticesIds();

        float accAngle = 0;
        int count = 0;
        for (int i = 0; i < positions.size(); i += 3)
        {
            Vec3f vert_ini(positions[i], positions[i + 1], positions[i + 2]);
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
    Frame frame_;
    Mesh mesh_;
    Vec2f global_exposure_;
    float global_scale_;
};

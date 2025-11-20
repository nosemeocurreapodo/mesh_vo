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
    KeyFrame(const Frame &frame, MeshCPU &mesh, float global_scale) : frame_(frame), mesh_(mesh)
    {
        global_scale_ = global_scale;
    }

    KeyFrame(const KeyFrame &other) : frame_(other.frame_), mesh_(other.mesh_), global_scale_(other.global_scale_)
    {
    }

    KeyFrame &operator=(const KeyFrame &other)
    {
        if (this != &other)
        {
            frame_ = other.frame_;
            mesh_ = other.mesh_;
            global_scale_ = other.global_scale_;
        }
        return *this;
    }

    const Frame &frame()
    {
        return frame_;
    }

    const MeshCPU &mesh() const
    {
        return mesh_;
    }

    MeshCPU &mesh()
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
        Vec2 localExpToGlobal(Vec2 localExp)
        {
            Vec2 globalExp;
            float alpha1 = std::exp(localExp(0));
            float alpha2 = std::exp(globalExp_(0));
            globalExp(0) = std::log(alpha1 / alpha2);
            globalExp(1) = localExp(1) - globalExp_(1) / alpha1;
            return globalExp;
        }

        Vec2 globalExpToLocal(Vec2 globalExp)
        {
            Vec2 localExp;
            float alpha1 = std::exp(-globalExp(0));
            float alpha2 = std::exp(-globalExp_(0));
            localExp(0) = std::log(alpha1 / alpha2);
            localExp(1) = -globalExp(1) + (alpha2 / alpha1) * globalExp_(1);
            return _localExp;
        }
        */

    /*
    void scaleVerticesAndWeights(float scale)
    {
        globalScale_ *= scale;
        geometry.scaleVertices(scale);
        // add a bit more of uncertanty to the weights
        // geometry.scaleWeights(scale * 1.2);
    }
    */

    float meanViewAngle(const SE3f &pose1, const SE3f &pose2)
    {
        int lvl = 1;

        // geometryType scene1 = geometry;
        // scene1.transform(pose1);
        //  scene1.project(cam);

        // geometryType scene2 = geometry;
        // scene2.transform(pose2);
        //  scene2.project(cam);

        /*
        auto pos_mm = mesh_.MapReadPositions();

        SE3f relativePose = pose1 * pose2.inverse();

        SE3f frame1PoseInv = relativePose.inverse();
        SE3f frame2PoseInv = SE3f();

        Vec3f frame1Translation = frame1PoseInv.translation();
        Vec3f frame2Translation = frame2PoseInv.translation();

        // std::vector<int> vIds = scene2.getVerticesIds();

        float accAngle = 0;
        int count = 0;
        for (int i = 0; i < mesh_.vertex_count(); i++)
        {
            Vec3f vert_ini(pos_mm[3 * i], pos_mm[3 * i + 1], pos_mm[3 * i + 2]);
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
        */
        return 1.0;
    }

private:
    Frame frame_;
    MeshCPU mesh_;
    float global_scale_;
};

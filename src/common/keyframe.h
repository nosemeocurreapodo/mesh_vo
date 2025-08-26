#pragma once

#include "params.h"
#include "core/types.h"
#include "core/camera.h"
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
    KeyFrame(const Frame frame, MeshCPU mesh) : frame_(frame), mesh_(mesh)
    {
    }

    KeyFrame(const KeyFrame &other) : frame_(other.frame_), mesh_(other.mesh_)
    {
    }

    KeyFrame &operator=(const KeyFrame &other)
    {
        if (this != &other)
        {
            frame_ = other.frame_;
            mesh_ = other.mesh_;
        }
        return *this;
    }

    const Frame &frame()
    {
        return frame_;
    }

    const MeshCPU &mesh()
    {
        return mesh_;
    }

    SE3 localPoseToGlobal(SE3 localPose)
    {
        SE3 localPoseScaled = localPose;
        //localPoseScaled.translation() /= globalScale_;
        SE3 globalPose = localPoseScaled * frame_.global_pose();
        return globalPose;
    }

    SE3 globalPoseToLocal(SE3 globalPose)
    {
        SE3 localPose = globalPose * frame_.global_pose().inverse();
        //localPose.translation() *= globalScale_;
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

    /*
        float meanViewAngle(SE3f pose1, SE3f pose2)
        {
            int lvl = 1;

            geometryType scene1 = geometry;
            scene1.transform(pose1);
            // scene1.project(cam);

            geometryType scene2 = geometry;
            scene2.transform(pose2);
            // scene2.project(cam);

            SE3f relativePose = pose1 * pose2.inverse();

            SE3f frame1PoseInv = relativePose.inverse();
            SE3f frame2PoseInv = SE3f();

            vec3f frame1Translation = frame1PoseInv.translation();
            vec3f frame2Translation = frame2PoseInv.translation();

            std::vector<int> vIds = scene2.getVerticesIds();

            float accAngle = 0;
            int count = 0;
            for (int vId : vIds)
            {
                vertex vert = scene2.getVertex(vId);

                vec3f diff1 = vert.ver - frame1Translation;
                vec3f diff2 = vert.ver - frame2Translation;

                assert(diff1.norm() > 0 && diff2.norm() > 0);

                vec3f diff1Normalized = diff1 / diff1.norm();
                vec3f diff2Normalized = diff2 / diff2.norm();

                float cos_angle = diff1Normalized.dot(diff2Normalized);
                cos_angle = std::clamp(cos_angle, -1.0f, 1.0f);
                float angle = std::acos(cos_angle);

                assert(!std::isnan(angle));

                accAngle += angle;
                count += 1;
            }

            return accAngle / count;
        }
    */
private:
    Frame frame_;
    MeshCPU mesh_;
};

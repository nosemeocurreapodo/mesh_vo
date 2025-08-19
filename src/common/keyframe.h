#pragma once

#include "params.h"
#include "common/types.h"
#include "common/camera.h"
#include "cpu/GeometryMesh.h"
#include "dataCPU.h"

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
    KeyFrame(const ImageType *im, int id, Vec2 globalExp, SE3 globalPose, float globalScale) : frame(im, id, localPose, globalPose)
    {
        globalScale_ = globalScale;
        // pose = SIM3f(scale, p.unit_quaternion(), p.translation());
    }

    KeyFrame(const KeyFrame &other)
    {
        frame_ = other.frame_;
        mesh_ = other.mesh_;
        globalScale_ = other.globalScale_;
    }

    KeyFrame &operator=(const KeyFrame &other)
    {
        if (this != &other)
        {
            frame_ = other.frame_;
            mesh_ = other.mesh_;
            globalScale_ = other.globalScale_;
        }
        return *this;
    }

    void initGeometryRandom(cameraType cam)
    {
        std::vector<vec2f> texcoords = uniformTexCoords();
        std::vector<vec3f> rays;
        std::vector<float> depths;
        std::vector<float> weights;
        for (vec2f pix : texcoords)
        {
            vec3f ray = cam.pixToRay(pix);
            rays.push_back(ray);
            float depth = randomDepth(mesh_vo::mapping_mean_depth * 0.5, mesh_vo::mapping_mean_depth * 1.5);
            depths.push_back(depth);
            weights.push_back(1.0 / mesh_vo::mapping_param_initial_var);
        }

        geometry.init(rays, depths, weights, cam);
    }

    void initGeometryVerticallySmooth(cameraType cam)
    {
        std::vector<vec2f> texcoords = uniformTexCoords();
        std::vector<vec3f> rays;
        std::vector<float> depths;
        std::vector<float> weights;
        for (vec2f pix : texcoords)
        {
            rays.push_back(cam.pixToRay(pix));
            float depth = verticallySmoothDepth(pix, mesh_vo::mapping_mean_depth * 0.5, mesh_vo::mapping_mean_depth * 1.5);
            depths.push_back(depth);
            weights.push_back(1.0 / mesh_vo::mapping_param_initial_var);
        }

        geometry.init(rays, depths, weights, cam);
    }

    void initGeometryFromDepth(const dataCPU<float> &depth, const dataCPU<float> &weight, cameraType cam)
    {
        std::vector<vec2f> texcoords = uniformTexCoords();
        std::vector<float> depths;
        std::vector<float> weights;

        std::vector<vec3f> rays;
        std::vector<vec2f> pixsWithNoData;

        for (vec2f texcoord : texcoords)
        {
            float dph = depth.get(texcoord(1), texcoord(0));
            float wght = weight.get(texcoord(1), texcoord(0));

            // assert(idph != idepth.nodata);
            if (dph == depth.nodata)
            {
                pixsWithNoData.push_back(texcoord);
                continue;
            }
            if (wght == weight.nodata)
            {
                wght = 1.0 / mesh_vo::mapping_param_initial_var;
            }

            assert(dph > 0.0);
            assert(!std::isnan(dph));
            assert(!std::isinf(dph));

            vec3f ray = cam.pixToRay(texcoord);

            rays.push_back(ray);
            depths.push_back(dph);
            weights.push_back(wght);

            assert(depths.size() <= mesh_vo::max_vertex_size);
        }

        geometry.init(rays, depths, weights, cam);

        vec2f minMax = geometry.minMaxDepthVertices();

        for (vec2f pixWithNoData : pixsWithNoData)
        {
            float depth = getDepthFromClosestShape(pixWithNoData, cam);
            // float depth = verticallySmoothDepth(texWithNoData, fromParamToDepth(MIN_PARAM), fromParamToDepth(MAX_PARAM), cam);
            // float depth = 1.0; // randomDepth(minMax(0), minMax(1));
            assert(depth > 0.0);

            // if(depth < fromParamToDepth(MIN_PARAM))
            //     depth = fromParamToDepth(MIN_PARAM);
            // if(depth > fromParamToDepth(MAX_PARAM))
            //     depth = fromParamToDepth(MAX_PARAM);

            float weight = 1.0 / mesh_vo::mapping_param_initial_var;

            vec3f ray = cam.pixToRay(pixWithNoData);
            rays.push_back(ray);
            depths.push_back(depth);
            weights.push_back(weight);
        }

        geometry.init(rays, depths, weights, cam);
    }

    SE3f localPoseToGlobal(SE3f _localPose)
    {
        SE3f localPoseScaled = _localPose;
        localPoseScaled.translation() /= globalScale;
        SE3f _globalPose = localPoseScaled * globalPose;
        return _globalPose;
    }

    SE3f globalPoseToLocal(SE3f _globalPose)
    {
        SE3f _localPose = _globalPose * globalPose.inverse();
        _localPose.translation() *= globalScale;
        return _localPose;
    }

    vec2f localExpToGlobal(vec2f _localExp)
    {
        vec2f _globalExp;
        float alpha1 = std::exp(_localExp(0));
        float alpha2 = std::exp(globalExp(0));
        _globalExp(0) = std::log(alpha1 / alpha2);
        _globalExp(1) = _localExp(1) - globalExp(1) / alpha1;
        return _globalExp;
    }

    vec2f globalExpToLocal(vec2f _globalExp)
    {
        vec2f _localExp;
        float alpha1 = std::exp(-_globalExp(0));
        float alpha2 = std::exp(-globalExp(0));
        _localExp(0) = std::log(alpha1 / alpha2);
        _localExp(1) = -_globalExp(1) + (alpha2 / alpha1) * globalExp(1);
        return _localExp;
    }

    void scaleVerticesAndWeights(float scale)
    {
        globalScale *= scale;
        geometry.scaleVertices(scale);
        // add a bit more of uncertanty to the weights
        // geometry.scaleWeights(scale * 1.2);
    }

    float getGlobalScale()
    {
        return globalScale;
    }

    void setGlobalExp(vec2f newGlobalExp)
    {
        globalExp = newGlobalExp;
    }

    vec2f getGlobalExp()
    {
        return globalExp;
    }

    void setGlobalPose(SE3f newGlobalPose)
    {
        globalPose = newGlobalPose;
    }

    SE3f getGlobalPose()
    {
        return globalPose;
    }

    dataCPU<imageType> &getRawImage(int lvl)
    {
        return raw_image.get(lvl);
    }

    dataCPU<vec2f> &getdIdpixImage(int lvl)
    {
        return dIdpix_image.get(lvl);
    }

    geometryType &getGeometry()
    {
        return geometry;
    }

    void setGeometry(const geometryType &_geometry)
    {
        geometry = _geometry;
    }

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

private:
    std::vector<vec2f> uniformTexCoords()
    {
        std::vector<vec2f> texcoords;
        for (float y = 0.0; y < mesh_vo::mesh_height; y++)
        {
            for (float x = 0.0; x < mesh_vo::mesh_width; x++)
            {
                vec2f pix;
                pix(0) = x / (mesh_vo::mesh_width - 1);
                pix(1) = y / (mesh_vo::mesh_height - 1);

                texcoords.push_back(pix);
            }
        }

        return texcoords;
    }

    float randomDepth(float min_depth, float max_depth)
    {
        float depth = (max_depth - min_depth) * float(rand() % 1000) / 1000.0 + min_depth;
        return depth;
    }

    float verticallySmoothDepth(vec2f pix, float min_depth, float max_depth)
    {
        // max depth when y = 0
        float depth = max_depth + (min_depth - max_depth) * pix(1);
        return depth;
    }

    float getDepthFromClosestShape(vec2f texcoord, cameraType cam)
    {
        std::vector<int> s_ids = geometry.getShapesIds();

        /*
        float best_depth = -1;
        float best_distance = 2.0;
        for(int s_id : s_ids)
        {
            shapeType shape = geometry.getShape(s_id);
            float distance = (texcoord - shape.getCenterPix()).norm();
            if(distance < best_distance)
            {
                best_distance = distance;
                shape.usePixel(texcoord);
                best_depth = shape.getDepth();
            }
        }
        if(best_depth > 0.0)
            return best_depth;
        else
            return mesh_vo::mapping_mean_depth;
        */

        float sigma = 0.1;

        float depth_sum = 0.0;
        float weight_sum = 0.0;
        for (int s_id : s_ids)
        {
            shapeType shape = geometry.getShape(s_id);
            float distance = (texcoord - shape.getCenterPix()).norm();
            float weight = std::exp(-(distance * distance) / (2 * sigma * sigma));

            shape.usePixel(texcoord);
            float depth = shape.getDepth();
            if (depth <= 0.0)
                continue;
            depth_sum += depth * weight;
            weight_sum += weight;
        }

        return depth_sum / weight_sum;
    }

    Frame frame_;
    MeshCPU mesh_;

    float globalScale_;
};

#pragma once

#include "params.h"
#include "common/types.h"
#include "common/camera.h"
#include "cpu/GeometryMesh.h"
#include "dataCPU.h"

class keyFrameCPU
{
public:
    keyFrameCPU(int width, int height)
        : raw_image(width, height, imageType(-1.0f)),
          dIdpix_image(width, height, vec2f(0.0f, 0.0f))
    {
        globalPose = SE3f();
        globalExp = {0.0f, 0.0f};
        globalScale = 1.0;
    };

    keyFrameCPU(const dataCPU<imageType> &im, const dataCPU<vec2f> &dIdpix, vec2f _globalExp, SE3f _globalPose, float _globalScale) : raw_image(im),
                                                                                                                                      dIdpix_image(dIdpix)
    {
        // raw_image.get(0) = im;
        // raw_image.generateMipmaps();

        // for (int lvl = 0; lvl < raw_image.getLvls(); lvl++)
        //{
        //     computeFrameDerivative(lvl);
        // }

        globalExp = _globalExp;
        globalPose = _globalPose;
        globalScale = _globalScale;
        // pose = SIM3f(scale, p.unit_quaternion(), p.translation());
    }

    keyFrameCPU(const keyFrameCPU &other)
        : raw_image(other.raw_image),
          dIdpix_image(other.dIdpix_image)
    {
        globalPose = other.globalPose;
        globalExp = other.globalExp;
        globalScale = other.globalScale;
        geometry = other.geometry;
    }

    keyFrameCPU &operator=(const keyFrameCPU &other)
    {
        if (this != &other)
        {
            globalPose = other.globalPose;
            globalExp = other.globalExp;
            raw_image = other.raw_image;
            dIdpix_image = other.dIdpix_image;
            globalScale = other.globalScale;
            geometry = other.geometry;
        }
        return *this;
    }

    void init(dataCPU<imageType> &im, vec2f _globalExp, SE3f _globalPose, float _globalScale)
    {
        assert(im.width == raw_image.get(0).width && im.height == raw_image.get(0).height);

        raw_image.get(0) = im;
        raw_image.generateMipmaps();

        for (int lvl = 0; lvl < raw_image.getLvls(); lvl++)
        {
            computeFrameDerivative(lvl);
        }

        globalExp = _globalExp;
        globalPose = _globalPose;
        globalScale = _globalScale;
        // pose = SIM3f(scale, p.unit_quaternion(), p.translation());
    }

    void initGeometryRandom(cameraType cam)
    {
        std::vector<vec3f> rays = uniformRays(cam);
        std::vector<float> depths;
        std::vector<float> weights;
        for (vec3f ray : rays)
        {
            float depth = randomDepth(0.5, 1.5);
            depths.push_back(depth);
            weights.push_back(1.0 / mesh_vo::mapping_param_initial_var);
        }

        geometry.init(rays, depths, weights, cam);
    }

    void initGeometryVerticallySmooth(cameraType cam)
    {
        std::vector<vec3f> rays = uniformRays(cam);
        std::vector<float> depths;
        std::vector<float> weights;
        for (vec3f ray : rays)
        {
            vec2f pix = cam.rayToPix(ray);
            float depth = verticallySmoothDepth(pix, 0.5, 1.5);
            depths.push_back(depth);
            weights.push_back(1.0 / mesh_vo::mapping_param_initial_var);
        }

        geometry.init(rays, depths, weights, cam);
    }

    void initGeometryFromDepth(dataCPU<float> &depth, dataCPU<float> &weight, cameraType cam)
    {
        std::vector<vec3f> rays = uniformRays(cam);
        std::vector<float> depths;
        std::vector<float> weights;

        std::vector<vec3f> raysWithData;
        std::vector<vec3f> raysWithNoData;

        for (vec3f ray : rays)
        {
            vec2f texcoord = cam.rayToPix(ray);
            float dph = depth.get(texcoord(1), texcoord(0));
            float wght = weight.get(texcoord(1), texcoord(0));

            // assert(idph != idepth.nodata);
            if (dph == depth.nodata)
            {
                raysWithNoData.push_back(ray);
                continue;
            }
            if (wght == weight.nodata)
            {
                wght = 1.0 / mesh_vo::mapping_param_initial_var;
            }

            assert(dph > 0.0);
            assert(!std::isnan(dph));
            assert(!std::isinf(dph));

            raysWithData.push_back(ray);
            depths.push_back(dph);
            weights.push_back(wght);

            assert(depths.size() <= mesh_vo::max_vertex_size);
        }

        geometry.init(raysWithData, depths, weights, cam);

        vec2f minMax = geometry.minMaxDepthVertices();

        for (vec3f rayWithNoData : raysWithNoData)
        {
            // float depth = getDepthFromClosestShape(texWithNoData, cam);
            // float depth = verticallySmoothDepth(texWithNoData, fromParamToDepth(MIN_PARAM), fromParamToDepth(MAX_PARAM), cam);
            float depth = 1.0; // randomDepth(minMax(0), minMax(1));
            assert(depth > 0.0);

            // if(depth < fromParamToDepth(MIN_PARAM))
            //     depth = fromParamToDepth(MIN_PARAM);
            // if(depth > fromParamToDepth(MAX_PARAM))
            //     depth = fromParamToDepth(MAX_PARAM);

            float weight = 1.0 / mesh_vo::mapping_param_initial_var;

            raysWithData.push_back(rayWithNoData);
            depths.push_back(depth);
            weights.push_back(weight);
        }

        geometry.init(raysWithData, depths, weights, cam);
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

    dataCPU<float> &getRawImage(int lvl)
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

    void setGeometry(geometryType _geometry)
    {
        geometry = _geometry;
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

    std::vector<vec3f> uniformRays(cameraType cam)
    {
        cameraParamType camParams = cam.getParams();

        std::vector<vec3f> rays;
        for (float y = 0.0; y < mesh_vo::mesh_height; y++)
        {
            for (float x = 0.0; x < mesh_vo::mesh_width; x++)
            {
                vec2f pix;
                pix(0) = x / (mesh_vo::mesh_width - 1);
                pix(1) = y / (mesh_vo::mesh_height - 1);

                vec3f ray = cam.pixToRay(pix);

                rays.push_back(ray);
            }
        }

        return rays;
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
        depth = std::clamp(depth, min_depth, max_depth);
        return depth;
    }

    float getDepthFromClosestShape(vec2f texcoord, cameraType cam)
    {
        std::vector<int> s_ids = geometry.getShapesIds();

        /*
        float best_depth = -1;
        float best_distance = cam.width*2;
        for(int s_id : s_ids)
        {
            shapeType shape = geometry.getShape(s_id);
            float distance = (texcoord - shape.getCenterPix()).norm();
            if(distance < best_distance)
            {
                best_distance = distance;
                best_depth = shape.getDepth(texcoord);
            }
        }

        return best_depth;
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

    void computeFrameDerivative(int lvl)
    {
        // dx.set(dx.nodata, lvl);
        // dy.set(dy.nodata, lvl);

        dataCPU<imageType> image = raw_image.get(lvl);

        int width = image.width;
        int height = image.height;

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                if (y == 0 || y == height - 1 || x == 0 || x == width - 1)
                {
                    // dx.set(0.0, y, x, lvl);
                    // dy.set(0.0, y, x, lvl);
                    dIdpix_image.get(lvl).setTexel(vec2f(0.0f, 0.0f), y, x);
                    continue;
                }

                float _dx = (float(image.getTexel(y, x + 1)) - float(image.getTexel(y, x - 1))) * width / 2.0;
                float _dy = (float(image.getTexel(y + 1, x)) - float(image.getTexel(y - 1, x))) * height / 2.0;

                dIdpix_image.get(lvl).setTexel(vec2f(_dx, _dy), y, x);
                // dx.set(_dx, y, x, lvl);
                // dy.set(_dy, y, x, lvl);
            }
    }

    dataMipMapCPU<imageType> raw_image;
    dataMipMapCPU<vec2f> dIdpix_image;
    geometryType geometry;
    float globalScale;

    SE3f globalPose;
    vec2f globalExp;
};

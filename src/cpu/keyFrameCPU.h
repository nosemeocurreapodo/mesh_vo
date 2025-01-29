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
        : raw_image(width, height, imageType(-1.0f))
    {
        globalPose = SE3f();
        globalExp = {0.0f, 0.0f};
        globalScale = 1.0;
    };

    keyFrameCPU(const keyFrameCPU &other)
        : raw_image(other.raw_image)
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
            globalScale = other.globalScale;
            geometry = other.geometry;
        }
        return *this;
    }

    void init(dataCPU<imageType> &im, vec2f _globalExp, SE3f _globalPose, float _globalScale)
    {
        raw_image.get(0) = im;
        raw_image.generateMipmaps();

        globalExp = _globalExp;
        globalPose = _globalPose;
        globalScale = _globalScale;
        // pose = SIM3f(scale, p.unit_quaternion(), p.translation());
    }

    void initGeometryRandom(camera cam)
    {
        std::vector<vec2f> texcoords = uniformTexCoords(cam);
        std::vector<float> depths;
        std::vector<float> weights;
        for (vec2f texcoord : texcoords)
        {
            float depth = randomDepth(fromParamToDepth(MIN_PARAM), fromParamToDepth(MAX_PARAM));
            depths.push_back(depth);
            weights.push_back(1.0 / (INITIAL_PARAM_STD * INITIAL_PARAM_STD));
        }

        geometry.init(texcoords, depths, weights, cam);
    }

    void initGeometryVerticallySmooth(camera cam)
    {
        std::vector<vec2f> texcoords = uniformTexCoords(cam);
        std::vector<float> depths;
        std::vector<float> weights;
        for (vec2f texcoord : texcoords)
        {
            float depth = verticallySmoothDepth(texcoord, fromParamToDepth(MIN_PARAM), fromParamToDepth(MAX_PARAM), cam);
            depths.push_back(depth);
            weights.push_back(1.0 / (INITIAL_PARAM_STD * INITIAL_PARAM_STD));
        }

        geometry.init(texcoords, depths, weights, cam);
    }

    void initGeometryFromDepth(dataCPU<float> &depth, dataCPU<float> &weight, camera cam)
    {
        assert(depth.width == cam.width && depth.height == cam.height);
        assert(weight.width == cam.width && weight.height == cam.height);

        std::vector<vec2f> texcoords = uniformTexCoords(cam);
        std::vector<float> depths;
        std::vector<float> weights;

        vec2f minMax = depth.getMinMax();
        assert(minMax(0) != depth.nodata && minMax(1) != depth.nodata && minMax(0) != minMax(1));

        std::vector<vec2f> texcoordsWithData;
        std::vector<vec2f> texcoordsWithNoData;

        for (vec2f texcoord : texcoords)
        {
            float dph = depth.get(texcoord(1), texcoord(0));
            float wght = weight.get(texcoord(1), texcoord(0));

            // assert(idph != idepth.nodata);
            if (dph == depth.nodata)
            {
                texcoordsWithNoData.push_back(texcoord);
                continue;
            }
            if (wght == weight.nodata)
            {
                wght = 1.0 / (INITIAL_PARAM_STD * INITIAL_PARAM_STD);
            }

            assert(dph > 0.0);
            assert(!std::isnan(dph));
            assert(!std::isinf(dph));

            texcoordsWithData.push_back(texcoord);
            depths.push_back(dph);
            weights.push_back(wght);

            assert(depths.size() <= MAX_VERTEX_SIZE);
        }

        geometry.init(texcoordsWithData, depths, weights, cam);

        for (vec2f texWithNoData : texcoordsWithNoData)
        {
            //float depth = getDepthFromClosestShape(texWithNoData, cam);
            float depth = verticallySmoothDepth(texWithNoData, fromParamToDepth(MIN_PARAM), fromParamToDepth(MAX_PARAM), cam);
            assert(depth > 0.0);

            //if(depth < fromParamToDepth(MIN_PARAM))
            //    depth = fromParamToDepth(MIN_PARAM);
            //if(depth > fromParamToDepth(MAX_PARAM))
            //    depth = fromParamToDepth(MAX_PARAM);

            float weight = 1.0 / (INITIAL_PARAM_STD * INITIAL_PARAM_STD);

            texcoordsWithData.push_back(texWithNoData);
            depths.push_back(depth);
            weights.push_back(weight);
        }

        geometry.init(texcoordsWithData, depths, weights, cam);
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
        //geometry.scaleWeights(scale * 1.2);
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

    geometryType &getGeometry()
    {
        return geometry;
    }

    void setGeometry(geometryType _geometry)
    {
        geometry = _geometry;
    }

private:
    std::vector<vec2f> uniformTexCoords(camera cam)
    {
        std::vector<vec2f> texcoords;
        for (float y = 0.0; y < MESH_HEIGHT; y++)
        {
            for (float x = 0.0; x < MESH_WIDTH; x++)
            {
                vec2f pix;
                pix(0) = (cam.width - 1) * x / (MESH_WIDTH - 1);
                pix(1) = (cam.height - 1) * y / (MESH_HEIGHT - 1);

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

    float verticallySmoothDepth(vec2f texcoord, float min_depth, float max_depth, camera cam)
    {
        //max depth when y = 0
        float depth = max_depth + (min_depth - max_depth) * texcoord(1) / (cam.width - 1.0);
        return depth;
    }

    float getDepthFromClosestShape(vec2f texcoord, camera cam)
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

        float sigma = 0.1*cam.width;

        float depth_sum = 0.0;
        float weight_sum = 0.0;
        for (int s_id : s_ids)
        {
            shapeType shape = geometry.getShape(s_id);
            float distance = (texcoord - shape.getCenterPix()).norm();
            float weight = std::exp(-(distance*distance) / (2*sigma*sigma));

            float depth = shape.getDepth(texcoord);
            if(depth <= 0.0)
                continue;
            depth_sum += depth*weight;
            weight_sum += weight;
        }

        return depth_sum/weight_sum;
    }

    dataMipMapCPU<imageType> raw_image;
    geometryType geometry;
    float globalScale;

    SE3f globalPose;
    vec2f globalExp;
};

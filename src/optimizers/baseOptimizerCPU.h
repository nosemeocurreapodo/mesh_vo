#pragma once

#include <Eigen/Core>
// #include <Eigen/CholmodSupport>
//  #include <Eigen/SPQRSupport>
// #include <thread>

#include "common/camera.h"
#include "common/types.h"
#include "common/DenseLinearProblem.h"
// #include "common/SparseLinearProblem.h"
//  #include "common/HGMapped.h"
#include "common/Error.h"
#include "common/common.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/renderCPU.h"
#include "cpu/reduceCPU.h"
// #include "cpu/SceneBase.h"
// #include "cpu/ScenePatches.h"
#include "cpu/SceneMesh.h"
// #include "cpu/SceneSurfels.h"
// #include "cpu/SceneMeshSmooth.h"
#include "cpu/OpenCVDebug.h"
#include "params.h"

template <typename sceneType>
class baseOptimizerCPU
{
public:
    baseOptimizerCPU(camera &_cam)
        : cam(_cam),
          image_buffer(_cam.width, _cam.height, -1.0),
          idepth_buffer(_cam.width, _cam.height, -1.0),
          error_buffer(_cam.width, _cam.height, -1.0),
          renderer(_cam.width, _cam.height)
    {
        cam = _cam;
    }

protected:
    void plotDebug(sceneType &scene, frameCPU &kframe, std::vector<frameCPU> &frames)
    {
        int lvl = 1;

        dataCPU<float> error_buffer(cam[lvl].width, cam[lvl].height, -1);
        dataCPU<float> idepth_buffer(cam[lvl].width, cam[lvl].height, -1);
        renderer.renderIdepthParallel(scene, kframe.getPose(), cam[lvl], idepth_buffer);

        show(kframe.getRawImage(lvl), "keyframe image", false);
        show(idepth_buffer, "kframe idepth", true);

        if (frames.size() > 0)
        {
            dataCPU<float> frames_buffer(cam[lvl].width * frames.size(), cam[lvl].height, -1);
            dataCPU<float> residual_buffer(cam[lvl].width * frames.size(), cam[lvl].height, -1);
            for (int i = 0; i < (int)frames.size(); i++)
            {
                error_buffer.set(error_buffer.nodata);
                renderer.renderResidualParallel(scene, kframe.getRawImage(lvl), kframe.getAffine(), kframe.getPose(), frames[i].getRawImage(lvl), frames[i].getAffine(), frames[i].getPose(), cam[lvl], error_buffer);

                for (int y = 0; y < cam[1].height; y++)
                {
                    for (int x = 0; x < cam[1].width; x++)
                    {
                        float pix_val = frames[i].getRawImage(lvl).get(y, x);
                        // float res_val = frames[i].getResidualImage().get(y, x, 1);
                        float res_val = error_buffer.get(y, x);

                        frames_buffer.set(pix_val, y, x + i * cam[lvl].width);
                        residual_buffer.set(res_val, y, x + i * cam[lvl].width);
                    }
                }
            }

            show(frames_buffer, "frames", false);
            show(residual_buffer, "residuals", false);
        }
    }

    Error computeError(frameCPU &frame, frameCPU &kframe, sceneType &scene, int lvl)
    {
        error_buffer.set(error_buffer.nodata, lvl);
        renderer.renderResidualParallel(scene, kframe.getRawImage(lvl), kframe.getAffine(), kframe.getPose(), frame.getRawImage(lvl), frame.getAffine(), frame.getPose(), cam[lvl], error_buffer.get(lvl));
        Error e = reducer.reduceErrorParallel(error_buffer.get(lvl));
        return e;
    }

    cameraMipMap cam;

    dataMipMapCPU<float> image_buffer;
    dataMipMapCPU<float> idepth_buffer;
    dataMipMapCPU<float> error_buffer;

    renderCPU<sceneType> renderer;
    reduceCPU reducer;
};

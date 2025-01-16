#pragma once

#include "params.h"
#include "common/camera.h"
#include "common/types.h"
#include "common/Error.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/renderCPU.h"
#include "cpu/reduceCPU.h"
#include "cpu/SceneMesh.h"
#include "common/DenseLinearProblem.h"
#include "cpu/OpenCVDebug.h"

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

        std::vector<dataCPU<float>> toShow;

        toShow.push_back(kframe.getRawImage(lvl));

        error_buffer.setToNoData(lvl);
        idepth_buffer.setToNoData(lvl);

        renderer.renderIdepthParallel(scene, kframe.getPose(), cam[lvl], idepth_buffer.get(lvl));

        toShow.push_back(idepth_buffer.get(lvl));

        for (int i = 0; i < (int)frames.size(); i++)
        {
            renderer.renderResidualParallel(scene, kframe.getRawImage(lvl), kframe.getExposure(), kframe.getPose(), frames[i].getRawImage(lvl), frames[i].getExposure(), frames[i].getPose(), cam[lvl], error_buffer.get(lvl));
            toShow.push_back(error_buffer.get(lvl));
        }

        show(toShow, "optimizer debug frames");
    }

    Error computeError(frameCPU &frame, frameCPU &kframe, sceneType &scene, int lvl)
    {
        error_buffer.set(error_buffer.nodata, lvl);
        renderer.renderResidualParallel(scene, kframe.getRawImage(lvl), kframe.getExposure(), kframe.getPose(), frame.getRawImage(lvl), frame.getExposure(), frame.getPose(), cam[lvl], error_buffer.get(lvl));
        Error e = reducer.reduceErrorParallel(error_buffer.get(lvl));
        return e;
    }

    cameraMipMap cam;

    dataMipMapCPU<float> image_buffer;
    dataMipMapCPU<float> idepth_buffer;
    dataMipMapCPU<float> error_buffer;

    renderCPU renderer;
    reduceCPU reducer;
};

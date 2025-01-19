#pragma once

#include "params.h"
#include "common/camera.h"
#include "common/types.h"
#include "common/Error.h"
#include "cpu/dataCPU.h"
#include "cpu/frameCPU.h"
#include "cpu/renderCPU.h"
#include "cpu/reduceCPU.h"
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
    void plotDebug(keyFrameCPU &kframe, std::vector<frameCPU> &frames, std::string window_name)
    {
        int lvl = 1;

        std::vector<dataCPU<float>> toShow;

        toShow.push_back(kframe.getRawImage(lvl));

        idepth_buffer.setToNoData(lvl);
        renderer.renderIdepthParallel(kframe, SE3f(), idepth_buffer, cam, lvl);

        toShow.push_back(idepth_buffer.get(lvl));

        for (int i = 0; i < (int)frames.size(); i++)
        {
            error_buffer.setToNoData(lvl);
            renderer.renderResidualParallel(kframe, frames[i], error_buffer, cam, lvl);
            toShow.push_back(error_buffer.get(lvl));
        }

        show(toShow, window_name);
    }

    Error computeError(frameCPU &frame, keyFrameCPU &kframe, int lvl)
    {
        error_buffer.setToNoData(lvl);
        renderer.renderResidualParallel(kframe, frame, error_buffer, cam, lvl);
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

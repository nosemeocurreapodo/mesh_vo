#include "optimizers/baseOptimizerCPU.h"

baseOptimizerCPU::baseOptimizerCPU(int width, int height)
    : image_buffer(width, height, -1.0),
      depth_buffer(width, height, -1.0),
      error_buffer(width, height, -1.0),
      weight_buffer(width, height, -1.0),
      renderer(width, height)
{
}

void baseOptimizerCPU::plotDebug(keyFrameCPU &kframe, std::vector<frameCPU> &frames, camera &cam, std::string window_name)
{
    int lvl = 1;

    std::vector<dataCPU<float>> toShow;

    toShow.push_back(kframe.getRawImage(lvl));

    depth_buffer.setToNoData(lvl);
    weight_buffer.setToNoData(lvl);

    renderer.renderDepthParallel(kframe, SE3f(), depth_buffer, cam, lvl);
    renderer.renderWeightParallel(kframe, SE3f(), weight_buffer, cam, lvl);

    depth_buffer.get(lvl).invert();

    toShow.push_back(depth_buffer.get(lvl));
    toShow.push_back(weight_buffer.get(lvl));

    for (int i = 0; i < (int)frames.size(); i++)
    {
        error_buffer.setToNoData(lvl);
        renderer.renderResidualParallel(kframe, frames[i], error_buffer, cam, lvl);
        // toShow.push_back(frames[i].getRawImage(lvl));
        toShow.push_back(error_buffer.get(lvl));
    }

    show(toShow, window_name);
}

Error baseOptimizerCPU::computeError(frameCPU &frame, keyFrameCPU &kframe, camera &cam, int lvl)
{
    error_buffer.setToNoData(lvl);
    renderer.renderResidualParallel(kframe, frame, error_buffer, cam, lvl);
    Error e = reducer.reduceErrorParallel(error_buffer.get(lvl));
    return e;
}
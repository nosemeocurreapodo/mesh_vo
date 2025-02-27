#include "optimizers/baseOptimizerCPU.h"

baseOptimizerCPU::baseOptimizerCPU(int width, int height)
    : image_buffer(width, height, -1.0),
      depth_buffer(width, height, -1.0),
      error_buffer(width, height, -1.0),
      weight_buffer(width, height, -1.0),
      renderer(width, height)
{
}

Error baseOptimizerCPU::computeError(frameCPU &frame, keyFrameCPU &kframe, cameraType &cam, int lvl)
{
    error_buffer.setToNoData(lvl);
    renderer.renderResidualParallel(kframe, frame, error_buffer, cam, lvl);
    Error e = reducer.reduceErrorParallel(error_buffer.get(lvl));
    return e;
}

bool baseOptimizerCPU::converged()
{
    return reachedConvergence;
}

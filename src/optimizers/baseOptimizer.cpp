#include "optimizers/baseOptimizerCPU.h"

BaseOptimizer::baseOptimizer(int width, int height)
    : image_buffer(width, height, 0),
      depth_buffer(width, height, -1.0),
      error_buffer(width, height, 0),
      weight_buffer(width, height, -1.0),
      renderer(width, height)
{
}

bool baseOptimizerCPU::converged()
{
    return reachedConvergence;
}

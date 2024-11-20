#include "optimizers/meshOptimizerCPU.h"

template class meshOptimizerCPU<SceneMesh, ShapeTriangleFlat, vec3<float>, vec3<int>>;
template class meshOptimizerCPU<ScenePatches, ShapePatch, vec1<float>, vec1<int>>;

template <typename sceneType, typename shapeType, typename jmapType, typename idsType>
void meshOptimizerCPU<sceneType, shapeType, jmapType, idsType>::initKeyframe(frameCPU &frame, int lvl)
{
    idepth_buffer.set(idepth_buffer.nodata, lvl);
    renderer.renderRandom(cam[lvl], &idepth_buffer, lvl);
    // renderer.renderSmooth(cam[lvl], &idepth_buffer, lvl, 0.5, 1.5);
    ivar_buffer.set(ivar_buffer.nodata, lvl);
    renderer.renderSmooth(cam[lvl], &ivar_buffer, lvl, initialIvar(), initialIvar());
    kscene.init(cam[lvl], idepth_buffer, ivar_buffer, lvl);
    kimage = frame.getRawImage();
    kpose = frame.getPose();
}
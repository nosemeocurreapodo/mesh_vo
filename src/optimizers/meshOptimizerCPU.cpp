#include "optimizers/meshOptimizerCPU.h"

template class meshOptimizerCPU<SceneMesh, ShapeTriangleFlat, vec3<float>, vec3<int>>;
template class meshOptimizerCPU<ScenePatches, ShapePatch, vec1<float>, vec1<int>>;

template <typename sceneType, typename shapeType, typename jmapType, typename idsType>
meshOptimizerCPU<sceneType, shapeType, jmapType, idsType>::meshOptimizerCPU(camera &_cam)
    : kimage(_cam.width, _cam.height, -1.0),
      image_buffer(_cam.width, _cam.height, -1.0),
      idepth_buffer(_cam.width, _cam.height, -1.0),
      ivar_buffer(_cam.width, _cam.height, -1.0),
      error_buffer(_cam.width, _cam.height, -1.0),
      jlightaffine_buffer(_cam.width, _cam.height, vec2<float>(0.0)),
      jpose_buffer(_cam.width, _cam.height, vec8<float>(0.0)),
      jmap_buffer(_cam.width, _cam.height, jmapType(0.0)),
      pId_buffer(_cam.width, _cam.height, idsType(-1)),
      debug(_cam.width, _cam.height, -1.0),
      idepthVar(_cam.width, _cam.height, -1.0),
      renderer(_cam.width, _cam.height)
{
    int lvl = 0;
    while(true)
    {
        camera lvlcam = _cam;
        lvlcam.resize(1.0 / std::pow(2.0, lvl));
        if(lvlcam.width == 0 || lvlcam.height == 0)
            break;
        cam.push_back(lvlcam);
        lvl++;
    }

    multiThreading = false;
    meshRegularization = 100.0;
    meshInitial = 0.0;
    kDepthAffine = vec2<float>(1.0, 0.0);
}

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
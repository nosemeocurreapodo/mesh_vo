#pragma once

#include "data_gpu.h"

class frame_gpu
{
public:

  frame_gpu();
  frame_gpu(int height, int width);

  data_gpu image;
  data_gpu der;
  data_gpu idepth;
  data_gpu error;
  data_gpu count;

  data_gpu jtra;
  data_gpu jrot;

  data_gpu gradient1;
  data_gpu gradient2;

  data_gpu hessian1;
  data_gpu hessian2;
  data_gpu hessian3;
  data_gpu hessian4;
  data_gpu hessian5;
  data_gpu hessian6;

  data_gpu jp0;
  data_gpu jp1;
  data_gpu jp2;

  Sophus::SE3f pose;

  bool init;

private:

};

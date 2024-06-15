#pragma once

#include "sophus/se3.hpp"

#include "data_cpu.h"
#include "params.h"

class frameCpu
{
public:
  frameCpu() : image(-1.0),
               dx(256.111111),
               dy(256.111111),
               idepth(-1.0),
               error(-1.0),
               count(-1.0)
  {
    init = false;
  };

  void copyTo(frameCpu &frame)
  {
    image.copyTo(frame.image);
    dx.copyTo(frame.dx);
    dy.copyTo(frame.dy);
    idepth.copyTo(frame.idepth);
    error.copyTo(frame.error);
    count.copyTo(frame.count);

    frame.pose = pose;
    frame.init = init;
  }

  data_cpu<float> image;
  data_cpu<float> dx;
  data_cpu<float> dy;
  data_cpu<float> idepth;
  data_cpu<float> error;
  data_cpu<int> count;

  Sophus::SE3f pose;

  bool init;
};

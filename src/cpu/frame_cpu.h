#pragma once

#include "sophus/se3.hpp"

#include "data_cpu.h"
#include "params.h"

class frameCpu
{
public:
  frameCpu() : image(CV_8UC1),
                der(CV_32FC2),
                idepth(CV_32FC1),
                error(CV_32FC1),
                count(CV_32SC1)
  {
    init = false;
  };

  void copyTo(frameCpu &frame)
  {
    image.copyTo(frame.image);
    der.copyTo(frame.der);
    idepth.copyTo(frame.idepth);
    error.copyTo(frame.error);
    count.copyTo(frame.count);

    frame.pose = pose;
    frame.init = init;
  }

  data_cpu image;
  data_cpu der;
  data_cpu idepth;
  data_cpu error;
  data_cpu count;

  Sophus::SE3f pose;

  bool init;
};

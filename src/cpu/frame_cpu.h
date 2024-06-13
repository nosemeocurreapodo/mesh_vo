#pragma once

#include <opencv2/core.hpp>
#include "sophus/se3.hpp"

#include "data_cpu.h"
#include "params.h"

class frame_cpu
{
public:
  frame_cpu();
  frame_cpu(int height, int width);

private:
  data_cpu image;
  data_cpu der;
  data_cpu idepth;
  data_cpu error;
  data_cpu count;

  Sophus::SE3f pose;

  bool init;
};

#pragma once

#include "sophus/se3.hpp"

#include <learnopengl/shader_m.h>

#include "data.h"

class frame
{
public:

  frame();
  frame(int height, int width);

  data image;
  data der;
  data idepth;
  Sophus::SE3f pose;

  bool init;

private:

};

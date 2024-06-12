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
  data error;
  data count;

  data jtra;
  data jrot;

  data gradient1;
  data gradient2;

  data hessian1;
  data hessian2;
  data hessian3;
  data hessian4;
  data hessian5;
  data hessian6;

  data jp0;
  data jp1;
  data jp2;

  Sophus::SE3f pose;

  bool init;

private:

};

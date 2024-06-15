#pragma once

#include "sophus/se3.hpp"

#include "data_cpu.h"
#include "params.h"

class frameCpu
{
public:
  frameCpu() : image(-1.0),
               dx(0.0),
               dy(0.0),
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

  void set(cv::Mat frame, cv::Mat id, Sophus::SE3f p = Sophus::SE3f())
  {
    set(frame, p);
    idepth.set(id);
    idepth.generateMipmaps();
  }

  void set(cv::Mat frame, Sophus::SE3f p = Sophus::SE3f())
  {
    image.set(frame);
    image.generateMipmaps();

    //computeFrameDerivative(0)
    //dx.generateMipmaps();
    //dy.generateMipmaps();

    for (int lvl = 0; lvl < MAX_LEVELS; lvl++)
    {
      computeFrameDerivative(lvl);
    }

    setRandomIdepth(0);
    idepth.generateMipmaps();

    pose = p;
    init = true;
  }

  void setRandomIdepth(int lvl)
  {
    idepth.set(idepth.nodata, lvl);

    for (int y = 0; y < idepth.sizes[lvl].height; y++) // los vertices estan en el medio del pixel!! sino puede agarrar una distancia u otra
    {
      for (int x = 0; x < idepth.sizes[lvl].width; x++)
      {
        float _idepth = 0.1 + (1.0 - 0.1) * float(y) / idepth.sizes[lvl].height;
        idepth.set(_idepth, y, x, lvl);
      }
    }
  }

  void computeFrameDerivative(int lvl)
  {
    dx.set(dx.nodata, lvl);
    dy.set(dy.nodata, lvl);

    for (int y = 1; y < image.sizes[lvl].height - 1; y++)
      for (int x = 1; x < image.sizes[lvl].width - 1; x++)
      {
        float _dx = (float(image.get(y, x + 1, lvl)) - float(image.get(y, x - 1, lvl))) / 2.0;
        float _dy = (float(image.get(y + 1, x, lvl)) - float(image.get(y - 1, x, lvl))) / 2.0;

        dx.set(_dx, y, x, lvl);
        dy.set(_dy, y, x, lvl);
      }
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

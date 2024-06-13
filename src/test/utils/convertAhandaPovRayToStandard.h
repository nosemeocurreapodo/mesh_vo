#pragma once

//#include "util/SophusUtil.h"
#include "sophus/se3.hpp"

Sophus::SE3f readPose(const char * filename);

#include "frame.h"

frame::frame()
{

}

frame::frame(int height, int width)
{
    image = data(height, width, 1, GL_UNSIGNED_BYTE);
    der   = data(height, width, 2, GL_FLOAT);
    idepth = data(height, width, 1, GL_FLOAT);
    pose = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero());
    init = false;
}




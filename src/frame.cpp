#include "frame.h"

frame::frame()
{

}

frame::frame(int height, int width)
{
    image = data(height, width, 1, GL_UNSIGNED_BYTE, GL_LINEAR_MIPMAP_NEAREST, GL_CLAMP_TO_BORDER);// GL_MIRRORED_REPEAT
    der   = data(height, width, 2, GL_FLOAT, GL_LINEAR_MIPMAP_NEAREST, GL_CLAMP_TO_BORDER);
    idepth = data(height, width, 1, GL_FLOAT, GL_LINEAR_MIPMAP_NEAREST, GL_CLAMP_TO_BORDER);
    pose = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero());
    init = false;
}




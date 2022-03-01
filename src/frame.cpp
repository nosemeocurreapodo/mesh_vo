#include "frame.h"

frame::frame()
{

}

frame::frame(int height, int width)
{
    image = data(height, width, 1, GL_UNSIGNED_BYTE, GL_NEAREST_MIPMAP_NEAREST,GL_MIRRORED_REPEAT);// GL_CLAMP_TO_BORDER);// GL_MIRRORED_REPEAT
    der   = data(height, width, 2, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    idepth = data(height, width, 1, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    pose = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero());
    vertexViewCount = Eigen::VectorXi(VERTEX_WIDTH*VERTEX_HEIGH);
    init = false;
}




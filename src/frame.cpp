#include "frame.h"

frame::frame()
{

}

frame::frame(int height, int width)
{
    image = data(height, width, 1, GL_UNSIGNED_BYTE, GL_LINEAR_MIPMAP_LINEAR,GL_MIRRORED_REPEAT);// GL_CLAMP_TO_BORDER);// GL_MIRRORED_REPEAT
    der   = data(height, width, 2, GL_FLOAT, GL_LINEAR_MIPMAP_LINEAR, GL_MIRRORED_REPEAT);
    idepth = data(height, width, 1, GL_FLOAT, GL_LINEAR_MIPMAP_LINEAR, GL_MIRRORED_REPEAT);
    error = data(height, width, 1, GL_FLOAT, GL_LINEAR_MIPMAP_LINEAR, GL_MIRRORED_REPEAT);
    count = data(height, width, 1, GL_FLOAT, GL_LINEAR_MIPMAP_LINEAR, GL_MIRRORED_REPEAT);

    jtra = data(height, width, 3, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    jrot = data(height, width, 3, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);

    gradient1 = data(height, width, 3, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    gradient2 = data(height, width, 3, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);

    hessian1 = data(height, width, 4, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    hessian2 = data(height, width, 4, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    hessian3 = data(height, width, 4, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    hessian4 = data(height, width, 4, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    hessian5 = data(height, width, 4, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    hessian6 = data(height, width, 4, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);

    jp0 = data(height, width, 1, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    jp1 = data(height, width, 1, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    jp2 = data(height, width, 1, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);

    pose = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero());
    init = false;
}





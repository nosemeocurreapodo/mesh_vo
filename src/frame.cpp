#include "frame.h"

struct HJPose{
    float J[6];
    float H[21];
    float error;
    int cout;
};

struct vec3{
    float x;
    float y;
    float z;
};

HJPose HJPoseCPUPerIndex(uchar *_frame, char* _frameDer, float* pose, uchar *_keyframe, float* idepth, float fx, float fy, float cx, float cy, float fxinv, float fyinv, float cxinv, float cyinv, int width, int height)
{
    HJPose _hjpose;

    for(int y = 0; y < height; y++)
        for(int x = 0; x < width; x++)
        {
            int kfindex = x + y*width;

            uchar vkf = _keyframe[kfindex];
            float keyframeId = idepth[kfindex];

            //std::cout << "keyframeId " << keyframeId << std::endl;

            if(keyframeId <= 0.0)
                continue;

            vec3 poinKeyframe;
            pointKeyframe.x = (fxinv[lvl]*x + cxinv[lvl])/keyframeId;
            pointKeyframe.y = (fyinv[lvl]*y + cyinv[lvl])/keyframeId;
            pointKeyframe.z = 1.0/keyframeId;

            vec3 pointFrame;
            pointFrame.x = relativePose[0]*poinKeyframe.x + relativePose[1]*poinKeyframe.y + relativePose[1]*poinKeyframe.z + relativePose[2];
            pointFrame.y = relativePose[3]*poinKeyframe.x + relativePose[4]*poinKeyframe.y + relativePose[5]*poinKeyframe.z + relativePose[6];
            pointFrame.z = relativePose[6]*poinKeyframe.x + relativePose[7]*poinKeyframe.y + relativePose[8]*poinKeyframe.z + relativePose[9];

            //std::cout << "pointFrame " << pointFrame << std::endl;

            if(pointFrame.z <= 0.0)
                continue;

            vec3 pixelFrame;
            pixelFrame.x = fx*pointFrame.x/pointFrame.z + cx;
            pixelFrame.y = fy*pointFrame.y/pointFrame.z + cy;
            pixelFrame.z = 1.0;

            //std::cout << "pixelFrame " << std::endl;

            if(pixelFrame.x < 0.0 || pixelFrame.x >= width || pixelFrame.y < 0.0 || pixelFrame.y >= height)
                continue;

            int findex = pixelFrame.x + pixelFrame.y*width;

            uchar vf = _frame[finxex];
            vec3 der;
            der.x =  _frameDer[findex];
            der.y = _frameDer[findex + width*height];

            //std::cout << "vf " << vf << " der " << der << std::endl;

            float id = 1.0/pointFrame.z;

            float v0 = der.x * fx * id;
            float v1 = der.y * fy * id;
            float v2 = -(v0 * pointFrame.x + v1 * pointFrame.y) * id;

            float J[6];
            J[0] = v0;
            J[1] = v1;
            J[2] = v2;
            J[3] = -pointFrame.z * v1 + pointFrame.y * v2;
            J[4] = pointFrame.z * v0 - pointFrame.x * v2;
            J[5] = -pointFrame.y * v0 + pointFrame.x * v1;

            float residual = (vf - vkf);
            _hjpose.error += residual*residual;
            _hjpose.count++;

            for(int i = 0; i < 6; i++)
            {
                _hjpose.J_pose(i) += J[i]*residual;
                for(int j = i; j < 6; j++)
                {
                    float jj = J[i]*J[j];
                    _hjpose.H_pose(i,j) += jj;
                    _hjpose.H_pose(j,i) += jj;
                }
            }
        }

    return _hjpose;
}


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

    jtra = data(height, width, 4, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    jrot = data(height, width, 4, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);

    gradient1 = data(height, width, 4, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);
    gradient2 = data(height, width, 4, GL_FLOAT, GL_NEAREST_MIPMAP_NEAREST, GL_MIRRORED_REPEAT);

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





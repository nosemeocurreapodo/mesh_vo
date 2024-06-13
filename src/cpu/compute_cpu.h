#pragma once

#include "frame_cpu.h"
#include "camera_cpu.h"
#include "HGPose_cpu.h"

#include "Utils/IndexThreadReduce.h"


class compute_cpu
{
public:
    compute_cpu()
    {

    };

    HGPose()

    show()

private:
    HGPose_cpu HGPosePerIndex(frame_cpu &_frame, frame_cpu &_keyframe, camera_cpu &_cam)

};


void compute_cpu::show(cv::Mat &_data, int lvl)
{
    cv::Mat toShow;
    cv::resize(_frame.cpuTexture[lvl],toShow,cv::Size(width[0],height[0]));
    cv::imshow("showCPU", toShow);
    cv::waitKey(30);
}

void mesh_vo::copyCPU(data &_src, data &_dst, int lvl)
{
    _src.cpuTexture[lvl].copyTo(_dst.cpuTexture[lvl]);
}

void compute_cpu::frameDerivative(frame_cpu &_frame, camera_cpu &_cam, int lvl)
{
    for(int y = 1; y < _cam.height[lvl]-1; y++)
        for(int x = 1; x < _cam.width[lvl]-1; x++)
        {
            cv::Vec2f d;
            d.val[0] = (_frame.image[lvl].at<uchar>(y,x+1) - _frame.image[lvl].at<uchar>(y,x-1))/2.0;
            d.val[1] = (_frame.image[lvl].at<uchar>(y+1,x) - _frame.image[lvl].at<uchar>(y-1,x))/2.0;

            _frame.der.cpuTexture[lvl].at<cv::Vec2f>(y,x) = d;
        }
}

HGPose_cpu compute_cpu::HGPose(frame_cpu &_frame, frame_cpu &_keyframe, camera_cpu &_cam, int lvl)
{
    HGPose_cpu _hgpose = HGPosePerIndex(_frame, _keyframe, _cam, lvl, 0, _cam.height[lvl]);
    //HJPose _hjpose = treadReducer.reduce(std::bind(&mesh_vo::HJPoseCPUPerIndex, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), _frame, lvl, 0, height[lvl]);
    _hjpose.H_pose /= _hjpose.count;
    _hjpose.J_pose /= _hjpose.count;
    _hjpose.error /= _hjpose.count;

    return _hjpose;
}

HGPose_cpu compute_cpu::HGPosePerIndex(frame_cpu &_frame, frame_cpu &_keyframe, camera &_camera, int lvl, int ymin, int ymax)
{
    HGPose_cpu _hgpose;

    Sophus::SE3f relativePose = _frame.pose*_keyframe.pose.inverse();

    for(int y = ymin; y < ymax; y++)
        for(int x = 0; x < width[lvl]; x++)
        {
            uchar vkf = keyframeData.image.cpuTexture[lvl].at<uchar>(y,x);
            float keyframeId = keyframeData.idepth.cpuTexture[lvl].at<float>(y,x);

            //std::cout << "keyframeId " << keyframeId << std::endl;

            if(keyframeId <= 0.0)
                continue;

            Eigen::Vector3f poinKeyframe = Eigen::Vector3f(fxinv[lvl]*x + cxinv[lvl],fyinv[lvl]*y + cyinv[lvl],1.0)/key/frameId;
            Eigen::Vector3f pointFrame = relativePose*poinKeyframe;

            //std::cout << "pointFrame " << pointFrame << std::endl;

            if(pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector3f pixelFrame = Eigen::Vector3f(fx[lvl]*pointFrame(0)/pointFrame(2) + cx[lvl], fy[lvl]*pointFrame(1)/pointFrame(2) + cy[lvl], 1.0);

            //std::cout << "pixelFrame " << std::endl;

            if(pixelFrame(0) < 0.0 || pixelFrame(0) >= width[lvl] || pixelFrame(1) < 0.0 || pixelFrame(1) >= height[lvl])
                continue;

            uchar vf = _frame->image.cpuTexture[lvl].at<uchar>(pixelFrame(1), pixelFrame(0));
            cv::Vec2f der = _frame->der.cpuTexture[lvl].at<cv::Vec2f>(pixelFrame(1),pixelFrame(0));

            Eigen::Vector2f d_f_d_uf(der.val[0],der.val[1]);

            //std::cout << "vf " << vf << " der " << der << std::endl;

            float id = 1.0/pointFrame(2);

            float v0 = d_f_d_uf(0) * fx[lvl] * id;
            float v1 = d_f_d_uf(1) * fy[lvl] * id;
            float v2 = -(v0 * pointFrame(0) + v1 * pointFrame(1)) * id;

            Eigen::Vector3f d_I_d_tra = Eigen::Vector3f(v0, v1, v2);
            Eigen::Vector3f d_I_d_rot = Eigen::Vector3f( -pointFrame(2) * v1 + pointFrame(1) * v2, pointFrame(2) * v0 - pointFrame(0) * v2, -pointFrame(1) * v0 + pointFrame(0) * v1);

            float residual = (vf - vkf);
            _hjpose.error += residual*residual;

            Eigen::Matrix<float, 6, 1> J;
            J << d_I_d_tra(0), d_I_d_tra(1), d_I_d_tra(2), d_I_d_rot(0), d_I_d_rot(1), d_I_d_rot(2);

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

float compute_cpu::error(frame_cpu *_frame, frame_cpu *_keyframe, camera_cpu *_cam, int lvl)
{
    float error = errorPerIndex(_frame, lvl, 0, height[lvl]);
    //float error = treadReducer.reduce(std::bind(&mesh_vo::errorCPUPerIndex, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), _frame, lvl, 0, height[lvl]);
    return _hjpose.error/_hjpose.count;
}

float compute_cpu::errorPerIndex(frame_cpu *_frame, frame_cpu *_keyframe, camera_cpu *_cam, int lvl, int ymin, int ymax)
{
    //std::cout << "entrando calcResidual" << std::endl;

    float error = 0.0;
    float count = 0.0;

    Sophus::SE3f relativePose = _frame->pose*_keyframe->pose.inverse();

    for(int y = ymin; y < ymax; y++)
        for(int x = 0; x < _cam->width[lvl]; x++)
        {
            uchar vkf = _keyframe->image[lvl].at<uchar>(y,x);
            float keyframeId = _keyframe->idepth[lvl].at<float>(y,x);

            if(keyframeId <= 0.0)
                continue;

            Eigen::Vector3f poinKeyframe = Eigen::Vector3f(_cam->fxinv[lvl]*(x+0.5) + _cam->cxinv[lvl], _cam->fyinv[lvl]*(y+0.5) + _cam->cyinv[lvl],1.0)/keyframeId;
            Eigen::Vector3f pointFrame = relativePose*poinKeyframe;

            //std::cout << "pointKeyframe " << poinKeyframe << std::endl;
            //std::cout << "pointFrame " << pointFrame << std::endl;

            if(pointFrame(2) <= 0.0)
                continue;

            Eigen::Vector3f pixelFrame = Eigen::Vector3f(_cam->fx[lvl]*pointFrame(0)/pointFrame(2) + _cam->cx[lvl], _cam->fy[lvl]*pointFrame(1)/pointFrame(2) + _cam->cy[lvl], 1.0);

            if(pixelFrame(0) < 0.0 || pixelFrame(0) > _cam->width[lvl] || pixelFrame(1) < 0 || pixelFrame(1) > _cam->height[lvl])
                continue;

            //std::cout << "pixelFrame " << pixelFrame << std::endl;
            uchar vf = _frame->image[lvl].at<uchar>(pixelFrame(1), pixelFrame(0));

            float residual = float(vf) - float(vkf);
            float error = residual*residual;

            _frame->error[lvl].at<float>(y,x) = error;

            error += error;
            count++;
        }

    if(count > 0)
    {
        error /= count;
    }

    return error;
}

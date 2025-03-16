#pragma once

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <iostream>
#include <chrono>

#include "cpu/dataCPU.h"
#include "common/types.h"
#include "common/camera.h"
#include "convertAhandaPovRayToStandard.h"

inline std::string &ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not_fn([](int ch)
                                                                    { return std::isspace(ch); })));
    return s;
}

inline std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not_fn([](int ch)
                                                           { return std::isspace(ch); }))
                .base(),
            s.end());
    return s;
}

inline std::string &trim(std::string &s)
{
    return ltrim(rtrim(s));
}

inline int getdir(std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL)
    {
        std::string name = std::string(dirp->d_name);

        if (name != "." && name != "..")
            files.push_back(name);
    }
    closedir(dp);

    std::sort(files.begin(), files.end());

    if (dir.at(dir.length() - 1) != '/')
        dir = dir + "/";
    for (unsigned int i = 0; i < files.size(); i++)
    {
        if (files[i].at(0) != '/')
            files[i] = dir + files[i];
    }

    return files.size();
}

inline int getFile(std::string source, std::vector<std::string> &files)
{
    std::ifstream f(source.c_str());

    if (f.good() && f.is_open())
    {
        while (!f.eof())
        {
            std::string l;
            std::getline(f, l);

            l = trim(l);

            if (l == "" || l[0] == '#')
                continue;

            files.push_back(l);
        }

        f.close();

        size_t sp = source.find_last_of('/');
        std::string prefix;
        if (sp == std::string::npos)
            prefix = "";
        else
            prefix = source.substr(0, sp);

        for (unsigned int i = 0; i < files.size(); i++)
        {
            if (files[i].at(0) != '/')
                files[i] = prefix + "/" + files[i];
        }

        return (int)files.size();
    }
    else
    {
        f.close();
        return -1;
    }
}

inline int getFilesAndTimestamps(std::string dir, std::string file, std::vector<std::string> &files, std::vector<double> &timestamps)
{
    std::ifstream f((dir + file).c_str());

    if (f.good() && f.is_open())
    {
        std::vector<std::string> lines;

        while (!f.eof())
        {
            std::string l;
            std::getline(f, l);

            l = trim(l);

            if (l == "" || l[0] == '#')
                continue;

            lines.push_back(l);
        }

        f.close();

        for (std::string line : lines)
        {
            std::stringstream ss(line);
            std::string timestamp_token;
            std::string filepath_token;
            std::getline(ss, timestamp_token, ' ');
            std::getline(ss, filepath_token, ' ');

            std::string file_path = dir + "/" + filepath_token;

            std::ifstream _f(file_path.c_str());

            if (_f.good())
            {
                files.push_back(file_path);
                timestamps.push_back(std::stod(timestamp_token));
            }
        }

        return files.size();
    }
    else
    {
        f.close();
        return -1;
    }
}

inline int getPosesAndTimestamps(std::string dir, std::string file, std::vector<SE3f> &poses, std::vector<double> &timestamps)
{
    std::ifstream f((dir + file).c_str());

    if (f.good() && f.is_open())
    {
        std::vector<std::string> lines;

        while (!f.eof())
        {
            std::string l;
            std::getline(f, l);

            l = trim(l);

            if (l == "" || l[0] == '#')
                continue;

            lines.push_back(l);
        }

        f.close();

        for (std::string line : lines)
        {
            std::stringstream ss(line);
            std::string token;
            std::vector<double> values;
            while (std::getline(ss, token, ' '))
            {
                values.push_back(std::stod(token));
            }

            SE3f pose;
            pose.setQuaternion(Eigen::Quaternionf(values[7], values[4], values[5], values[6]));
            pose.translation() = Eigen::Vector3f(values[1], values[2], values[3]);

            poses.push_back(pose);
            timestamps.push_back(values[0]);
        }

        return poses.size();
    }
    else
    {
        f.close();
        return -1;
    }
}

inline SE3f getClosestPose(std::vector<SE3f> poses, std::vector<double> timestamps, double target_timestamp)
{
    SE3f closest_pose;
    double closest_diff = 10000000000.0;
    for (int i = 0; i < poses.size(); i++)
    {
        double timestamp = timestamps[i];
        SE3f pose = poses[i];

        double diff = std::abs(timestamp - target_timestamp);
        if (diff < closest_diff)
        {
            closest_pose = pose;
            closest_diff = diff;
        }
    }
    return closest_pose;
}

// Function to compute error between two SE3 poses
inline float computeSE3Error(const SE3f &pose_est, const SE3f &pose_gt)
{
    // Compute the relative transformation: error transformation T_error
    SE3f T_error = pose_est.inverse() * pose_gt;

    // Convert T_error to a 6D vector (Lie algebra) representing the error
    vec6f error_vector = T_error.log();

    // Return the norm of the error vector
    return error_vector.norm();
}

// Function to compute error between two SE3 poses
inline float computeImageError(const dataCPU<float> &image_est, const dataCPU<float> &image_gt)
{
    assert(image_est.width == image_gt.width && image_est.height == image_gt.height);

    float error = 0.0;
    int count = 0;
    for (int y = 0; y < image_est.height; y++)
    {
        for (int x = 0; x < image_est.width; x++)
        {
            float est = image_est.getTexel(y, x);
            float gt = image_gt.getTexel(y, x);
            if (est != image_est.nodata && gt != image_gt.nodata)
            {
                error += (est - gt) * (est - gt);
                count += 1;
            }
        }
    }
    return error / count;
}

class load_dataset_base
{
public:
    load_dataset_base()
    {
    }

    std::vector<std::string> getImageFiles()
    {
        return image_files;
    }

    std::vector<std::string> getDepthFiles()
    {
        return depth_files;
    }

    std::vector<SE3f> getPoses()
    {
        return poses;
    }

    std::vector<double> getTimestamps()
    {
        return timestamps;
    }

    cameraType getCamera()
    {
        return cam;
    }

    int getWidth()
    {
        return w;
    }

    int getHeight()
    {
        return h;
    }

    float getDepthFactor()
    {
        return depthFactor;
    }

protected:
    int w, h;
    float depthFactor;

    std::vector<std::string> image_files;
    std::vector<std::string> depth_files;
    std::vector<SE3f> poses;
    std::vector<double> timestamps;
    cameraType cam;
};

class load_dataset_tum_rgbd : public load_dataset_base
{
public:
    load_dataset_tum_rgbd()
        : load_dataset_base()
    {
        std::string dataset_path = std::string(TEST_DATA_DIR) + "/rgbd_dataset_freiburg1_floor_part";
        std::string image_path = "/rgb.txt";
        std::string depth_path = "/depth.txt";
        std::string pose_path = "/groundtruth.txt";

        // for tum rgbd dataset
        w = 640;
        h = 480;
        depthFactor = 5000.0;

        float fx = 525.0;
        float fy = 525.0;
        float cx = 319.5;
        float cy = 239.5;

        cam = cameraType(fx, fy, cx, cy, w, h);

        // std::string poses_path = std::string(TEST_DATA_DIR) + dataset_path + pose_path;
        // poses = getPosesFromFile(poses_path);

        std::vector<std::string> image_file_paths;
        std::vector<double> image_timestamps;
        getFilesAndTimestamps(dataset_path, image_path, image_file_paths, image_timestamps);

        std::vector<std::string> depth_file_paths;
        std::vector<double> depth_timestamps;
        getFilesAndTimestamps(dataset_path, depth_path, depth_file_paths, depth_timestamps);

        std::vector<SE3f> poses_list;
        std::vector<double> pose_timestamps;
        getPosesAndTimestamps(dataset_path, pose_path, poses_list, pose_timestamps);

        std::vector<SE3f> sync_poses;
        for (double timestamp : image_timestamps)
        {
            sync_poses.push_back(getClosestPose(poses_list, pose_timestamps, timestamp));
        }

        image_files = image_file_paths;
        depth_files = depth_file_paths;
        poses = sync_poses;
        timestamps = image_timestamps;
    }

private:
};

class load_dataset_icl_nuim : public load_dataset_base
{
public:
    load_dataset_icl_nuim()
        : load_dataset_base()
    {
        std::string dataset_path = std::string(TEST_DATA_DIR) + "/traj3n_frei_png_part";
        std::string assosiations_path = dataset_path + "/associations.txt";
        std::string pose_path = dataset_path + "/traj3n.gt.freiburg";

        w = 640;
        h = 480;
        depthFactor = 5000.0;

        float fx = 481.20;
        float fy = -480.0;
        float cx = 319.5;
        float cy = 239.5;

        cam = cameraType(fx, fy, cx, cy, w, h);

        readAssociationsFile(dataset_path, assosiations_path, image_files, depth_files);

        
    }

private:
    int readAssociationsFile(std::string dir, std::string file, std::vector<std::string> &image_files, std::vector<std::string> &depth_files)
    {
        std::ifstream f((dir + file).c_str());

        if (f.good() && f.is_open())
        {
            std::vector<std::string> lines;

            while (!f.eof())
            {
                std::string l;
                std::getline(f, l);

                l = trim(l);

                if (l == "" || l[0] == '#')
                    continue;

                lines.push_back(l);
            }

            f.close();

            for (std::string line : lines)
            {
                std::stringstream ss(line);
                std::string index_token;
                std::string depth_token;
                std::string image_token;
                std::getline(ss, index_token, ' ');
                std::getline(ss, depth_token, ' ');
                std::getline(ss, image_token, ' ');

                std::string image_path = dir + "/" + image_token;
                std::string depth_path = dir + "/" + depth_token;

                std::ifstream _f(image_path.c_str());

                if (_f.good())
                {
                    image_files.push_back(image_path);
                    depth_files.push_back(depth_path);
                }
            }

            return image_files.size();
        }
        else
        {
            f.close();
            return -1;
        }
    }
};

/*
void use_test_dataset()
{
    dataset_path = "/test";
    image_path = "/rgb";
    depth_path = "/depth";
    pose_path = "/poses";

    w = 640;
    h = 480;
    depthFactor = 10.0;

    float fx = 481.20;
    float fy = 480.0;
    float cx = 319.5;
    float cy = 239.5;

    cam = cameraType(fx, fy, cx, cy, w, h);

    std::string poses_path = std::string(TEST_DATA_DIR) + dataset_path + pose_path;
    std::vector<std::string> pose_files;
    getdir(poses_path, pose_files);
    for (std::string pose_file : pose_files)
    {
        SE3f pose = readPovRaypose(pose_file);
        poses.push_back(pose);
    }
}
*/
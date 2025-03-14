#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <iostream>
#include <chrono>

#include "cpu/dataCPU.h"
#include "common/types.h"

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

inline std::vector<SE3f> getPoses(std::string source)
{
    std::ifstream f(source.c_str());

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

        std::vector<SE3f> poses;

        for (std::string line : lines)
        {
            std::stringstream ss(line);
            std::string token;
            std::vector<float> values;
            while (std::getline(ss, token, ' '))
            {
                values.push_back(std::stof(token));
            }
            SE3f pose;
            pose.setQuaternion(Eigen::Quaternionf(values[4], values[5], values[6], values[7]));
            pose.translation() = Eigen::Vector3f(values[1], values[2], values[3]);
            poses.push_back(pose);
        }

        return poses;
    }
    else
    {
        f.close();
        return std::vector<SE3f>();
    }
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

void load_icl_nuim_dataset()
{
    std::string dataset_path = "/traj3n_frei_png_part";
    std::string image_path = "/rgb";
    std::string depth_path = "/depth";
    std::string pose_path = "/traj3n.gt.freiburg";

    // for icl-nuim dataset
    int w = 640;
    int h = 480;
    float fx = 481.20;
    float fy = -480.0;
    float cx = 319.5;
    float cy = 239.5;
}

void rgbd_dataset()
{
    std::string dataset_path = "/rgbd_dataset_freiburg1_floor_part";
    std::string image_path = "/rgb";
    std::string depth_path = "/depth";
    std::string pose_path = "/groundtruth.txt";

    // for tum rgbd dataset
    int w = 640;
    int h = 480;
    float fx = 525.0;
    float fy = 525.0;
    float cx = 319.5;
    float cy = 239.5;
}

class load_dataset
{
public:

    load_dataset(std::string dataset_name)
    {
        if(dataset_name == "icl-nuim")
        {
            load_icl_nuim_dataset();
        }
        else if(dataset_name == "tum-rgbd")
        {
            rgbd_dataset();
        }
        else
        {
            std::cout << "Invalid dataset name" << std::endl;
        }
    }
    // open image files: first try to open as file.
    // std::string images_path = std::string(TEST_DATA_DIR) + "/traj3n_frei_png_part/rgb";
    std::string images_path = std::string(TEST_DATA_DIR) + dataset_path + image_path;

    std::vector<std::string> image_files;

    if (getdir(images_path, image_files) >= 0)
    {
        printf("found %d image files in folder %s!\n", (int)image_files.size(), images_path.c_str());
    }
    else
    {
        printf("could not load file list! wrong path / file?\n");
    }

    std::vector<std::string> depth_files;
    std::string depths_path = std::string(TEST_DATA_DIR) + dataset_path + depth_path;

    if (getdir(depths_path, depth_files) >= 0)
    {
        printf("found %d image files in folder %s!\n", (int)depth_files.size(), depths_path.c_str());
    }
    else
    {
        printf("could not load file list! wrong path / file?\n");
    }

    std::string poses_path = std::string(TEST_DATA_DIR) + dataset_path + pose_path;
    std::vector<SE3f> poses = getPoses(poses_path);

    cameraType cam(fx, fy, cx, cy, w, h);


private:

    std::string dataset_path, image_path;
    std::string image_path;
    std::string depth_path;
    std::string pose_path;

    int w, h;
    float fx, fy, cx, cy;
}
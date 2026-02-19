#pragma once

// #include <sstream>
#include <fstream>
#include <dirent.h>
// #include <algorithm>
// #include <cctype>
#include <string>
// #include <iostream>
// #include <chrono>

// #include "core/types.h"
#include "mpdr/common/camera.h"

// inline std::string &ltrim(std::string &s)
//{
//     s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not_fn([](int ch)
//                                                                     { return std::isspace(ch); })));
//     return s;
// }

inline std::string &ltrim(std::string &s)
{
    s.erase(s.begin(),
            std::find_if(s.begin(), s.end(), [](unsigned char ch)
                         { return !std::isspace(ch); }));
    return s;
}

// inline std::string &rtrim(std::string &s)
//{
//     s.erase(std::find_if(s.rbegin(), s.rend(), std::not_fn([](int ch)
//                                                            { return std::isspace(ch); }))
//                 .base(),
//             s.end());
//     return s;
// }

inline std::string &rtrim(std::string &s)
{
    s.erase(
        std::find_if(s.rbegin(), s.rend(),
                     [](unsigned char ch)
                     { return !std::isspace(ch); })
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

class LoadDatasetBase
{
public:
    LoadDatasetBase(double fx, double fy, double cx, double cy, int w, int h)
        : w_(w), h_(h), cam_(fx, fy, cx, cy, w, h)
    {
    }

    std::vector<std::string> GetImageFiles()
    {
        return image_files_;
    }

    std::vector<std::string> GetDepthFiles()
    {
        return depth_files_;
    }

    std::vector<SE3<float>> GetPoses()
    {
        return poses_;
    }

    std::vector<double> GetTimestamps()
    {
        return time_stamps_;
    }

    PinholeCamera<float> GetCamera()
    {
        return cam_;
    }

    unsigned int GetWidth()
    {
        return w_;
    }

    unsigned int GetHeight()
    {
        return h_;
    }

    double GetDepthFactor()
    {
        return depth_factor_;
    }

protected:
    SE3<float> GetClosestPose(const std::vector<SE3<float>> &poses, const std::vector<double> &time_stamps, double target_timestamp)
    {
        SE3<float> closest_pose;
        double closest_diff = 10000000000.0;
        for (size_t i = 0; i < poses.size(); i++)
        {
            double time_stamp = time_stamps[i];
            SE3<float> pose = poses[i];

            double diff = std::abs(time_stamp - target_timestamp);
            if (diff < closest_diff)
            {
                closest_pose = pose;
                closest_diff = diff;
            }
        }
        return closest_pose.inverse();
    }

    unsigned int w_, h_;
    double depth_factor_;

    std::vector<std::string> image_files_;
    std::vector<std::string> depth_files_;
    std::vector<SE3<float>> poses_;
    std::vector<double> time_stamps_;
    PinholeCamera<float> cam_;
};

class LoadDatasetTumRgbd : public LoadDatasetBase
{
public:
    LoadDatasetTumRgbd(std::string test_data_path)
        : LoadDatasetBase(525.0, 525.0, 319.5, 239.5, 640, 480) // for tum rgbd dataset
    {
        std::string dataset_path = test_data_path + "/rgbd_dataset_freiburg1_floor_part";
        std::string image_path = "/rgb.txt";
        std::string depth_path = "/depth.txt";
        std::string pose_path = "/groundtruth.txt";

        this->depth_factor_ = 5000.0 / 255.0;

        // std::string poses_path = test_data_path + dataset_path + pose_path;
        // poses = getPosesFromFile(poses_path);

        std::vector<std::string> image_file_paths;
        std::vector<double> image_timestamps;
        GetFilesAndTimestamps(dataset_path, image_path, image_file_paths, image_timestamps);

        std::vector<std::string> depth_file_paths;
        std::vector<double> depth_timestamps;
        GetFilesAndTimestamps(dataset_path, depth_path, depth_file_paths, depth_timestamps);

        std::vector<SE3<float>> poses_list;
        std::vector<double> pose_timestamps;
        GetPosesAndTimestamps(dataset_path, pose_path, poses_list, pose_timestamps);

        std::vector<SE3<float>> sync_poses;
        for (double timestamp : image_timestamps)
        {
            sync_poses.push_back(GetClosestPose(poses_list, pose_timestamps, timestamp));
        }

        this->image_files_ = image_file_paths;
        this->depth_files_ = depth_file_paths;
        this->time_stamps_ = image_timestamps;
        this->poses_ = sync_poses;
    }

private:
    inline int GetFilesAndTimestamps(std::string dir, std::string file, std::vector<std::string> &files, std::vector<double> &timestamps)
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
            throw std::runtime_error("Failed to open file");
        }
    }

    int GetPosesAndTimestamps(std::string dir, std::string file, std::vector<SE3<float>> &poses, std::vector<double> &timestamps)
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

                SE3<float> pose;
                pose.setQuaternion(Quaternion<float>(values[7], values[4], values[5], values[6]));
                pose.translation() = Vec3<float>(values[1], values[2], values[3]);

                poses.push_back(pose);
                timestamps.push_back(values[0]);
            }

            return poses.size();
        }
        else
        {
            throw std::runtime_error("Failed to open file");
        }
    }
};

class LoadDatasetIclNuim : public LoadDatasetBase
{
public:
    LoadDatasetIclNuim(std::string test_data_path)
        : LoadDatasetBase(481.20, -480.0, 319.5, 239.5, 640, 480)
    {
        std::string dataset_path = test_data_path + "/traj3_frei_png_part";
        std::string assosiations_path = "/associations.txt";
        std::string pose_path = "/traj3.gt.freiburg";

        this->depth_factor_ = 5000.0 / 255.0; // 275.0;

        ReadAssociationsFile(dataset_path, assosiations_path, this->image_files_, this->depth_files_, this->time_stamps_);

        std::vector<SE3<float>> pose_list;
        std::vector<double> poses_timestamps;
        GetPosesAndTimestamps(dataset_path, pose_path, pose_list, poses_timestamps);

        std::vector<SE3<float>> sync_poses;
        for (double time_stamp : this->time_stamps_)
        {
            sync_poses.push_back(this->GetClosestPose(pose_list, poses_timestamps, time_stamp));
        }

        this->poses_ = sync_poses;
    }

private:
    int ReadAssociationsFile(std::string dir, std::string file, std::vector<std::string> &image_files, std::vector<std::string> &depth_files, std::vector<double> &timestamps)
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
                std::vector<std::string> tokens;
                while (std::getline(ss, token, ' '))
                {
                    tokens.push_back(token);
                }

                double image_index = std::stod(tokens[0]);
                // double depth_index = std::stod(tokens[2]);
                std::string image_path = dir + "/" + tokens[3];
                std::string depth_path = dir + "/" + tokens[1];

                std::ifstream _f(image_path.c_str());
                std::ifstream __f(depth_path.c_str());

                if (_f.good() && __f.good())
                {
                    image_files.push_back(image_path);
                    depth_files.push_back(depth_path);
                    timestamps.push_back(image_index);
                }
            }

            return image_files.size();
        }
        else
        {
            throw std::runtime_error("Failed to open associations file");
        }
    }

    int GetPosesAndTimestamps(std::string dir, std::string file, std::vector<SE3<float>> &poses, std::vector<double> &timestamps)
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

                SE3<float> pose;
                pose.setQuaternion(Quaternion<float>(values[7], values[4], values[5], values[6]));
                pose.translation() = Vec3<float>(values[1], values[2], values[3]);

                // linalg::Mat4<float> pos2opencv = linalg::Mat4<float>::Identity();
                // pos2opencv(1, 1) = -1.0f;
                // pos2opencv(2, 2) = -1.0f;

                // linalg::Mat4<float> corrected = pos2opencv * pose.matrix();

                // linalg::SE3<float> posecorrected(corrected);

                poses.push_back(pose);
                timestamps.push_back(values[0]);
            }

            return poses.size();
        }
        else
        {
            throw std::runtime_error("Failed to open poses and timestamps file");
        }
    }
};

class LoadDesktopDataset : public LoadDatasetBase
{
public:
    LoadDesktopDataset(std::string test_data_path)
        : LoadDatasetBase(481.20, 480.0, 319.5, 239.5, 640, 480)
    {
        std::string dataset_path = test_data_path + "/desktop_dataset";
        std::string images_path = dataset_path + "/images";
        std::string depth_path = dataset_path + "/depths";
        std::string poses_path = dataset_path + "/poses";

        this->depth_factor_ = 5000.0 / 255.0; // 275.0;

        getdir(images_path, this->image_files_);
        getdir(depth_path, this->depth_files_);
        std::vector<std::string> pose_files;
        getdir(poses_path, pose_files);

        for (std::string pose_file : pose_files)
        {
            SE3<float> pose = readPose(pose_file);
            this->poses_.push_back(pose);
        }
    }

private:
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

    SE3<float> readPose(std::string filename)
    {
        std::ifstream cam_pars_file(filename);
        if (!cam_pars_file.is_open())
        {
            std::cout << "failed opening file! " + filename << std::endl;
            exit(1);
        }

        char readlinedata[300];

        Vec3<float> direction;
        Vec3<float> upvector;
        Vec3<float> posvector;

        while (1)
        {
            cam_pars_file.getline(readlinedata, 300);
            //         cout<<readlinedata<<endl;
            if (cam_pars_file.eof())
                break;

            std::istringstream iss;

            if (strstr(readlinedata, "cam_dir") != NULL)
            {

                std::string cam_dir_str(readlinedata);

                cam_dir_str = cam_dir_str.substr(cam_dir_str.find("= [") + 3);
                cam_dir_str = cam_dir_str.substr(0, cam_dir_str.find("]"));

                iss.str(cam_dir_str);
                iss >> direction(0); //.x();
                iss.ignore(1, ',');
                iss >> direction(2); //.z();
                iss.ignore(1, ',');
                iss >> direction(1); //.y();
                iss.ignore(1, ',');
            }

            if (strstr(readlinedata, "cam_up") != NULL)
            {

                std::string cam_up_str(readlinedata);

                cam_up_str = cam_up_str.substr(cam_up_str.find("= [") + 3);
                cam_up_str = cam_up_str.substr(0, cam_up_str.find("]"));

                iss.str(cam_up_str);
                iss >> upvector(0); //.x();
                iss.ignore(1, ',');
                iss >> upvector(2); //.z();
                iss.ignore(1, ',');
                iss >> upvector(1); //.y();
                iss.ignore(1, ',');
            }

            if (strstr(readlinedata, "cam_pos") != NULL)
            {
                //            cout<< "cam_pos is present!"<<endl;

                std::string cam_pos_str(readlinedata);

                cam_pos_str = cam_pos_str.substr(cam_pos_str.find("= [") + 3);
                cam_pos_str = cam_pos_str.substr(0, cam_pos_str.find("]"));

                //            cout << "cam pose str = " << endl;
                //            cout << cam_pos_str << endl;

                iss.str(cam_pos_str);
                iss >> posvector(0); //.x();
                iss.ignore(1, ',');
                iss >> posvector(2); //.z();
                iss.ignore(1, ',');
                iss >> posvector(1); //.y();
                iss.ignore(1, ',');
                //             cout << "position: "<<posvector.x<< ", "<< posvector.y << ", "<< posvector.z << endl;
            }
        }

        //    R=Mat(3,3,CV_64F);
        //    R.row(0)=Mat(direction.cross(upvector)).t();
        //    R.row(1)=Mat(-upvector).t();
        //    R.row(2)=Mat(direction).t();

        Mat3<float> Rot;
        // Rot.row(0) = (direction.cross(upvector)).transpose();
        // Rot.row(1) = (-upvector).transpose();
        // Rot.row(2) = direction.transpose();
        Vec3<float> row0 = (direction.cross(upvector));
        Vec3<float> row1 = -upvector;
        Vec3<float> row2 = direction;

        //row0 = row0.normalized();
        //row1 = row1.normalized();
        //row2 = row2.normalized();

        Rot(0, 0) = row0(0);
        Rot(0, 1) = row0(1);
        Rot(0, 2) = row0(2);
        Rot(1, 0) = row1(0);
        Rot(1, 1) = row1(1);
        Rot(1, 2) = row1(2);
        Rot(2, 0) = row2(0);
        Rot(2, 1) = row2(1);
        Rot(2, 2) = row2(2);

        // T=-R*Mat(posvector);

        Vec3<float> Tra = -Rot * posvector;
        Tra = Tra / 2000.0f;

        SE3<float> pose = SE3<float>(Rot, Tra);

        /*
        std::ofstream myfile;

        char new_filename[500];

        //file name
        sprintf(new_filename,"%s.new", filename);

        std::cout << "writing " << new_filename << std::endl;

        myfile.open(new_filename);
        //myfile << "Writing this to a file.\n";
        myfile << pose.matrix() << std::endl;
        myfile.close();
        */

        return pose;
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

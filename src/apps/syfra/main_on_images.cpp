/**
 * This file is part of LSD-SLAM.
 *
 * Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
 * For more information see <http://vision.in.tum.de/lsdslam>
 *
 * LSD-SLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * LSD-SLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with LSD-SLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <functional>

#include "opencv2/opencv.hpp"

#include "sophus/se3.hpp"

#include "common/camera.h"
#include "visualOdometry.h"

std::string &ltrim(std::string &s)
{
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not_fn([](int c) { return std::isspace(c); })));
	return s;
}
std::string &rtrim(std::string &s)
{
	s.erase(std::find_if(s.rbegin(), s.rend(), std::not_fn([](int c) { return std::isspace(c); })).base(), s.end());
	return s;
}
std::string &trim(std::string &s)
{
	return ltrim(rtrim(s));
}
int getdir(std::string dir, std::vector<std::string> &files)
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

int getFile(std::string source, std::vector<std::string> &files)
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

int main(int argc, char **argv)
{
	if(argc != 3)
    {
        std::cout << "usage: " << argv[0] << " /path/to/dataset initial_index" << std::endl;
        return 0;
    }

	int width = 1280;
	int height = 720;
	float fx = 0.594896555 * width;
	float fy = 1.10809731 * height;
	float cx = 0.500066221 * width;
	float cy = 0.491324276 * height;
	float k1 = -0.0853296965;

	cameraType cam(fx, fy, cx, cy, k1, width, height);

	visualOdometry odometry(cam, width, height);

	int initial_id = std::atoi(argv[2]);

	// open image files: first try to open as file.
	std::string source = argv[1];
	std::vector<std::string> files;

	if (getdir(source, files) >= 0)
	{
		printf("found %d image files in folder %s!\n", (int)files.size(), source.c_str());
	}
	else if (getFile(source, files) >= 0)
	{
		printf("found %d image files in file %s!\n", (int)files.size(), source.c_str());
	}
	else
	{
		printf("could not load file list! wrong path / file?\n");
	}

	// get HZ
	//double hz = std::atof(argv[2]);

	cv::Mat image = cv::Mat(height, width, CV_8U);
	int runningIDX = 0;
	float fakeTimeStamp = 0;

	for (unsigned int i = initial_id; i < files.size(); i++)
	{
		cv::Mat imageDist = cv::imread(files[i], cv::IMREAD_GRAYSCALE);

		if (imageDist.rows != height || imageDist.cols != width)
		{
			if (imageDist.rows * imageDist.cols == 0)
				printf("failed to load image %s! skipping.\n", files[i].c_str());
			else
				printf("image %s has wrong dimensions - expecting %d x %d, found %d x %d. Skipping.\n",
					   files[i].c_str(),
					   width, height, imageDist.cols, imageDist.rows);
			continue;
		}
		assert(imageDist.type() == CV_8U);

		image = imageDist;
		//undistorter->undistort(imageDist, image);
		assert(image.type() == CV_8U);

		// cv::imshow("image", image);
		// cv::waitKey(30);

		image.convertTo(image, CV_32FC1);

		dataCPU<float> imageData(width, height, -1.0);
		imageData.set((float *)image.data);

		if (runningIDX == 0)
		{
			odometry.init(imageData, SE3f());
		// system->randomInit(image.data, fakeTimeStamp, runningIDX);
		}
		else
			odometry.intrinsicAndLocAndMap(imageData);
		// system->trackFrame(image.data, runningIDX ,hz == 0,fakeTimeStamp);

		runningIDX++;
		fakeTimeStamp += 0.03;

		// if(hz != 0)
		//	r.sleep();

		/*
		if(fullResetRequested)
		{

			printf("FULL RESET!\n");
			delete system;

			system = new SlamSystem(w, h, K, doSlam);
			system->setVisualization(outputWrapper);

			fullResetRequested = false;
			runningIDX = 0;
		}

		ros::spinOnce();

		if(!ros::ok())
			break;
		*/
	}

	/*
	system->finalize();



	delete system;
	delete undistorter;
	delete outputWrapper;
	return 0;
	*/
}

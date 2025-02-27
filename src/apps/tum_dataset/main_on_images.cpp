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

#include "opencv2/opencv.hpp"
#include "Undistorter.h"

#include "visualOdometry.h"
#include "visualOdometryThreaded.h"

std::string &ltrim(std::string &s)
{
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not_fn([](int ch)
																	{ return std::isspace(ch); })));
	return s;
}
std::string &rtrim(std::string &s)
{
	s.erase(std::find_if(s.rbegin(), s.rend(), std::not_fn([](int ch)
														   { return std::isspace(ch); }))
				.base(),
			s.end());
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

using namespace lsd_slam;
int main(int argc, char **argv)
{
	if (argc != 5)
	{
		std::cout << "usage: " << argv[0] << " /path/to/calibfile /path/to/dataset start_index end_index" << std::endl;
		return 0;
	}

	// get camera calibration in form of an undistorter object.
	// if no undistortion is required, the undistorter will just pass images through.
	std::string calibFile = argv[1];
	Undistorter *undistorter = 0;

	undistorter = Undistorter::getUndistorterForFile(calibFile.c_str());

	if (undistorter == 0)
	{
		printf("need camera calibration file! (set using _calib:=FILE)\n");
		exit(0);
	}

	int w_inp = undistorter->getInputWidth();
	int h_inp = undistorter->getInputHeight();

	int w = undistorter->getOutputWidth();
	int h = undistorter->getOutputHeight();
	float fx = undistorter->getK().at<double>(0, 0);
	float fy = undistorter->getK().at<double>(1, 1);
	float cx = undistorter->getK().at<double>(2, 0);
	float cy = undistorter->getK().at<double>(2, 1);

	// int w = undistorter->getInputWidth();
	// int h = undistorter->getInputHeight();
	// float fx = undistorter->getOriginalK().at<double>(0, 0) * w;
	// float fy = undistorter->getOriginalK().at<double>(1, 1) * h;
	// float cx = undistorter->getOriginalK().at<double>(2, 0) * w;
	// float cy = undistorter->getOriginalK().at<double>(2, 1) * h;

	// Sophus::Matrix3f K;
	// K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;

	// open image files: first try to open as file.
	std::string source = argv[2];
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
	// double hz = std::atof(argv[2]);

	int start_index = std::atoi(argv[3]);
	int end_index = std::atoi(argv[4]);

	cv::Mat image = cv::Mat(h, w, CV_8U);
	int runningIDX = 0;
	float fakeTimeStamp = 0;

	cameraType cam(fx, fy, cx, cy, w, h);

	visualOdometryThreaded odometry(cam);

	for (unsigned int i = start_index+1; i < end_index; i++) // files.size()
	{
		cv::Mat imageDist = cv::imread(files[i], cv::IMREAD_GRAYSCALE);

		if (imageDist.rows != h_inp || imageDist.cols != w_inp)
		{
			if (imageDist.rows * imageDist.cols == 0)
				printf("failed to load image %s! skipping.\n", files[i].c_str());
			else
				printf("image %s has wrong dimensions - expecting %d x %d, found %d x %d. Skipping.\n",
					   files[i].c_str(),
					   w, h, imageDist.cols, imageDist.rows);
			continue;
		}
		assert(imageDist.type() == CV_8U);

		undistorter->undistort(imageDist, image);
		// image = imageDist;
		assert(image.type() == CV_8U);

		// cv::imshow("image", image);
		// cv::waitKey(30);

		image.convertTo(image, CV_32FC1);
		// cv::resize(image, image, cv::Size(cam.width, cam.height), cv::INTER_AREA);

		dataCPU<float> imageData(w, h, -1.0);
		imageData.set((float *)image.data);

		if (runningIDX == 0)
		{
			odometry.init(imageData, SE3f());
			// system->randomInit(image.data, fakeTimeStamp, runningIDX);
		}
		else
			odometry.locAndMap(imageData);
		// system->trackFrame(image.data, runningIDX ,hz == 0,fakeTimeStamp);

		runningIDX++;
		fakeTimeStamp += 0.03;

		std::this_thread::sleep_for(std::chrono::milliseconds(1000));

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

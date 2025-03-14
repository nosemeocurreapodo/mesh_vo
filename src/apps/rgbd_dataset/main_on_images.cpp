#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <iostream>

#include "opencv2/opencv.hpp"

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

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		std::cout << "usage: " << argv[0] << " /path/to/dataset" << std::endl;
		return 0;
	}

	int w = 640;
	int h = 480;
	float fx = 525.0;
	float fy = 525.0;
	float cx = 319.5;
	float cy = 239.5;

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
		printf("could not load file list from %s! wrong path / file?\n", source.c_str());
	}

	cameraType cam(fx, fy, cx, cy, w, h);

	visualOdometryThreaded odometry(w, h, true, true);

	for (unsigned int i = 0; i < files.size(); i++)
	{
		cv::Mat image = cv::imread(files[i], cv::IMREAD_GRAYSCALE);

		if (image.rows != h || image.cols != w)
		{
			if (image.rows * image.cols == 0)
				printf("failed to load image %s! skipping.\n", files[i].c_str());
			else
				printf("image %s has wrong dimensions - expecting %d x %d, found %d x %d. Skipping.\n",
					   files[i].c_str(),
					   w, h, image.cols, image.rows);
			continue;
		}

		if (std::is_same<imageType, uchar>::value)
			image.convertTo(image, CV_8UC1);
		else if (std::is_same<imageType, int>::value)
			image.convertTo(image, CV_32SC1);
		else if (std::is_same<imageType, float>::value)
			image.convertTo(image, CV_32FC1);

		dataCPU<imageType> imageData(w, h, 0);
		imageData.set((imageType *)image.data);

		if (i == 0)
		{
			odometry.init(imageData, SE3f(), cam);
		}
		else
			odometry.locAndMap(imageData);

		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
}

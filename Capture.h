#ifndef CAPTURE_CAPTURE_H
#define CAPTURE_CAPTURE_H

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class Capture {
public:
    explicit Capture(int camera);

    ~Capture();

    void run();

    cv::Mat frame;
    cv::Mat gray;
private:
    int camera;
    cv::VideoCapture cap;
    cv::Mat mapx, mapy;
    cv::Matx33d intrinsics;
    cv::Vec4d distortion_coeff;
};


#endif //CAPTURE_CAPTURE_H
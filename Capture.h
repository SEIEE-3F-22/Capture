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

    py::array_t<unsigned char> get();

private:
    cv::VideoCapture cap;
    cv::Mat mapx, mapy;

    bool newFrameReceived = false;

    cv::Mat corrected;

    static py::array_t<unsigned char> Mat2ndarray(cv::Mat src);
};


#endif //CAPTURE_CAPTURE_H
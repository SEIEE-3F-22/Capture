#ifndef CAPTURE_CAPTURE_H
#define CAPTURE_CAPTURE_H

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <condition_variable>

namespace py = pybind11;

class Capture {
public:
    explicit Capture(int camera);

    ~Capture();

    void Acquire();

    void Undistort();

    void Openvino();

    py::array_t<unsigned char> Get();

private:
    cv::VideoCapture cap;
    bool newFrameReceived = false;
    bool newFrameFinished = false;
    cv::Mat mapx, mapy;
    cv::Mat frame, corrected;

    std::mutex frame_mutex, corrected_mutex;
    std::condition_variable cv_frameReceived;

    static py::array_t<unsigned char> Mat2ndarray(const cv::Mat& src);
};


#endif //CAPTURE_CAPTURE_H
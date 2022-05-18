#ifndef CAPTURE_CAPTURE_H
#define CAPTURE_CAPTURE_H

#include <iostream>
#include <fstream>
#include <condition_variable>
#include <thread>
#include <mutex>
#include <utility>

#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "Openvino.h"

namespace py = pybind11;

class Capture {
public:
    explicit Capture(int camera, std::string input_model_path, std::string device_name);

    ~Capture();

    void Acquire();

    void Undistort();

    py::array_t<unsigned char> getImage();

    void beginInference();

    std::vector<Object> getInferenceResult();

private:
    cv::VideoCapture cap;
    bool newFrameReceived = false;
    bool newFrameFinished = false;
    cv::Mat mapx, mapy;
    cv::Mat frame, image;

    std::mutex frame_mutex, image_mutex;
    std::condition_variable frameReceived;

    CNNNetwork network;
    std::vector<Object> resultObjects;
    InferRequest infer_request;
    std::string input_name;
    std::string output_name;

    void initNetwork(std::string input_model_path, std::string device_name);

    static py::array_t<unsigned char> Mat2ndarray(const cv::Mat& src);
};


#endif //CAPTURE_CAPTURE_H
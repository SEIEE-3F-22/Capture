#ifndef CAPTURE_CAPTURE_H
#define CAPTURE_CAPTURE_H

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <condition_variable>

#include "Openvino.h"

namespace py = pybind11;

class Capture {
public:
    explicit Capture(int camera, std::string input_model_path, std::string device_name);

    ~Capture();

    void Acquire();

    void Undistort();

    cv::Mat getFrame();

    py::array_t<unsigned char> getCorrected();

    void OpenvinoInference();

    std::vector<Object> getInferenceResult();

private:
    cv::VideoCapture cap;
    bool newFrameReceived = false;
    bool newFrameFinished = false;
    cv::Mat mapx, mapy;
    cv::Mat frame, corrected;

    std::mutex frame_mutex, corrected_mutex;
    std::condition_variable cv_frameReceived;

    CNNNetwork network;
    std::vector<Object> objects;
    InferRequest infer_request;
    std::string input_name;
    std::string output_name;

    void OpenvinoInit(std::string input_model_path, std::string device_name);

    static py::array_t<unsigned char> Mat2ndarray(const cv::Mat& src);
};


#endif //CAPTURE_CAPTURE_H
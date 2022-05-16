#ifndef CAPTURE_OPENVINO_H
#define CAPTURE_OPENVINO_H

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <inference_engine.hpp>

using namespace InferenceEngine;

/**
 * @brief Define names based depends on Unicode path support
 */
#define tcout                  std::cout
#define file_name_t            std::string
#define imread_t               cv::imread
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.3
#define LOGD(fmt, ...) printf("[%s][%s][%d]: " fmt "\n", __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)

static const int INPUT_W = 416; //416
static const int INPUT_H = 416; //416
static const int NUM_CLASSES = 5; // COCO has 80 classes. Modify this value on your own dataset.

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
};

const float color_list[5][3] = //80
        {
                {0.000, 0.447, 0.741},
                {0.850, 0.325, 0.098},
                {0.929, 0.694, 0.125},
                {0.494, 0.184, 0.556},
                {0.466, 0.674, 0.188},
        };

#endif //CAPTURE_OPENVINO_H

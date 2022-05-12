#include "Capture.h"

Object Capture::getInferenceResult(){
    return objects;
}

void Capture::OpenvinoInit(std::string input_model, std::string device_name){
    network = ie.ReadNetwork(input_model);
    // 载模型
}

void Capture::OpenvinoInference(){
    cv::Mat image;
    while (true) {
        image = getFrame();
        // inference
    }
}
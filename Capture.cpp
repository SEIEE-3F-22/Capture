#include "Capture.h"
#include "Openvino.cpp"

Capture::Capture(int camera, std::string input_model_path, std::string device_name) {
    cap.open(camera);
    if (!cap.isOpened()) {
        throw std::runtime_error("Couldn't open video capture device");
    }
    std::cout << "-------Video capture device opened-------" << std::endl;

    std::cout << "-------Undistort Initialization started-------" << std::endl;

    cv::Matx33d intrinsics;
    cv::Vec4d distortion_coeff;

    std::ifstream intrinsicFile("intrinsics.txt");
    std::ifstream disFile("dis_coeff.txt");

    for (auto i = 0; i < 3; i++) {
        for (auto j = 0; j < 3; j++) {
            intrinsicFile >> intrinsics(i, j);
        }
    }
    std::cout << "Intrinsics:\r\n" << intrinsics << std::endl;
    for (auto i = 0; i < 4; i++) {
        disFile >> distortion_coeff(i);
    }
    std::cout << "Distortion:\r\n" << distortion_coeff << std::endl;
    cv::fisheye::initUndistortRectifyMap(intrinsics, distortion_coeff, cv::Matx33d::eye(), intrinsics,
                                         cv::Size(640, 480), CV_16SC2, mapx, mapy);
    std::cout << "-------Undistort Initialization finished-------" << std::endl;

    std::cout << "-------Openvino Initialization started-------" << std::endl;
    initNetwork(std::move(input_model_path), std::move(device_name));
    std::cout << "-------Openvino Initialization finished-------" << std::endl;
}

void Capture::Acquire() {
    while (true) {
        /** The mutex is available */
        if (frame_mutex.try_lock()) {
            cap >> frame;   //不判断图像有效性
            frame_mutex.unlock();
            if (!frame.empty()) {
                newFrameReceived = true;
                frameReceived.notify_all();
            }
        }

            /** The mutex is unavailable */
        else {
            cv::Mat tmpMat;
            cap >> tmpMat;
            if (!tmpMat.empty()) {
                std::unique_lock<std::mutex> lock(frame_mutex);
                frame = tmpMat.clone();
                newFrameReceived = true;
                frameReceived.notify_all();
                // The unique_lock decomposed here, release the frame_mutex
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void Capture::Undistort() {
    cv::Mat gray, corrected, binary, kernel;

    while (true) {
        do {
            std::unique_lock<std::mutex> lock(frame_mutex);
            frameReceived.wait(lock, [=]() { return newFrameReceived; });
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            // The unique_lock decomposed here, release the frame_mutex
        } while (false);
        newFrameReceived = false;

        do {
            std::unique_lock<std::mutex> lock(image_mutex);
            remap(gray, corrected, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
            cv::threshold(corrected, binary, 0, 255, cv::THRESH_OTSU);

            kernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
            morphologyEx(binary, image, cv::MORPH_OPEN, kernel);
            // The unique_lock decomposed here, release the image_mutex
        } while (false);
        newFrameFinished = true;
    }
}

py::array_t<unsigned char> Capture::getImage() {
    if (!newFrameFinished) {
        return Mat2ndarray(cv::Mat(0, 0, CV_8UC1));
    }
    std::unique_lock<std::mutex> lock(image_mutex);
    newFrameFinished = false;
    return Mat2ndarray(image);
    // The unique_lock decomposed here, release the image_mutex
}

py::array_t<unsigned char> Capture::Mat2ndarray(const cv::Mat &src) {
    py::array_t<unsigned char> dst = py::array_t<unsigned char>({src.rows, src.cols}, src.data);
    return dst;
}

Capture::~Capture() {
    cap.release();
}

Capture capture(1, "/home/pi/Code/Capture/test/4_yolox_nano.xml", "MYRIAD");

void captureAcquire() {
    capture.Acquire();
}

void captureUndistort() {
    capture.Undistort();
}

void captureInference() {
    capture.beginInference();
}

void captureRun() {
    std::ios::sync_with_stdio(false);

    std::thread thread_Acquire(captureAcquire);
    std::thread thread_Undistort(captureUndistort);
    std::thread thread_Openvino(captureInference);

    thread_Acquire.detach();
    thread_Undistort.detach();
    thread_Openvino.detach();
}

py::array_t<unsigned char> captureGet() {
    return capture.getImage();
}

std::vector<float> captureGetferenceResult() {
    std::vector<float> result;
    std::vector<Object> objects = capture.getInferenceResult();
    if (!objects.empty()) {
        for (int i = 0; i < objects.size(); ++i) {
            result.push_back(objects[i].label);
            result.push_back(objects[i].rect.x);
            result.push_back(objects[i].rect.y);
            result.push_back(objects[i].rect.width);
            result.push_back(objects[i].rect.height);
            result.push_back(objects[i].prob);
        }
    }
    return result;
}

PYBIND11_MODULE(Capture, m) {
    m.def("run", &captureRun, py::call_guard<py::gil_scoped_release>());
    m.def("get", &captureGet);
    m.def("getInferenceResult", &captureGetferenceResult, py::return_value_policy::reference);
}
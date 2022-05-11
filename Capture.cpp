#include "Capture.h"

Capture::Capture(int camera){

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
}

void Capture::run() {
    cv::Mat frame, gray;
    while (true) {
        cap >> frame;
        if (!frame.empty()) {
            newFrameReceived = true;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            remap(gray, corrected, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
        }
    }
}

py::array_t<unsigned char> Capture::get() {
    if(!newFrameReceived) {
        return Mat2ndarray(cv::Mat(0, 0, cv::CV_8UC1));
    }
    newFrameReceived = false;
    return Mat2ndarray(corrected);
}

py::array_t<unsigned char> Capture::Mat2ndarray(cv::Mat src) {
    py::array_t<unsigned char> dst = py::array_t<unsigned char>({src.rows, src.cols}, src.data);
    return dst;
}

Capture::~Capture() {
    cap.release();
}

Capture capture(0);

void captureRun() {
    capture.run();
}

py::array_t<unsigned char> captureGet() {
    return capture.get();
}

PYBIND11_MODULE(Capture, m) {
    m.def("run", &captureRun, py::call_guard<py::gil_scoped_release>());
    m.def("get", &captureGet);
}
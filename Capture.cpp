#include "Capture.h"

namespace pybind11 {
    namespace detail {
        template<>
        struct type_caster<cv::Mat> {
        public:
        PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));
            //! 1. cast numpy.ndarray to cv::Mat
            bool load(handle src, bool) {
                array_t<unsigned char> ndarray = reinterpret_borrow<array_t<unsigned char>>(src);
                py::buffer_info buf = ndarray.request();
                value = cv::Mat(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char *) buf.ptr);
                return true;
            }
            //! 2. cast cv::Mat to numpy.ndarray
            static handle cast(const cv::Mat &src, return_value_policy, handle) {
                py::array_t<unsigned char> dst = py::array_t<unsigned char>({src.rows, src.cols}, src.data);
                return dst;
            }
        };
    }
}//! end namespace pybind11::detail

Capture::Capture(int camera) : camera(camera) {
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
    std::cout << "Initialization finished" << std::endl;
}

void Capture::run() {
    cap.open(camera);
    if (!cap.isOpened()) {
        throw std::runtime_error("Couldn't open video capture device");
    }
    cv::Mat corrected;
    while (true) {
        cap >> frame;
        if (!frame.empty()) {
            remap(frame, corrected, mapx, mapy, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
            cv::cvtColor(corrected, gray, cv::COLOR_BGR2GRAY);
        }
    }
}

Capture::~Capture() {
    cap.release();
}

PYBIND11_MODULE(Capture, m) {
    py::class_<Capture>(m, "Capture")
            .def(py::init<int>())
            .def("run", &Capture::run)
            .def("get", [](Capture &self) {
                py::array_t<unsigned char> image = py::detail::type_caster<cv::Mat>::cast(self.gray);
                return image;
            });
}
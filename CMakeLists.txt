cmake_minimum_required(VERSION 3.16)
project(Capture)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a")
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/test)
message(Library will be generated under ${LIBRARY_OUTPUT_PATH})

set(INTEL_OPENVINO_DIR "/opt/intel/openvino_2021")
set(INTEL_CVSDK_DIR "/opt/intel/openvino_2021")
set(InferenceEngine_DIR "/opt/intel/openvino_2021/deployment_tools/inference_engine/share")

find_package(OpenCV REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(pybind11)

pybind11_add_module(Capture Capture.cpp)
target_link_libraries(
        Capture
        PUBLIC ${OpenCV_LIBS}
        PUBLIC ${InferenceEngine_LIBRARIES}
        PUBLIC ${NGRAPH_LIBRARIES}
)

cmake_minimum_required(VERSION 3.16)
project(Capture)

set(CMAKE_CXX_STANDARD 14)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/test)
message(Library will be generated under ${LIBRARY_OUTPUT_PATH}/test)

set(INTEL_OPENVINO_DIR "/opt/intel/openvino_2021")
set(INTEL_CVSDK_DIR "/opt/intel/openvino_2021")

find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(pybind11)

pybind11_add_module(Capture Capture.cpp Openvino.cpp)
target_link_libraries(Capture PUBLIC ${OpenCV_LIBS} ${InferenceEngine_LIBRARIES} ${NGRAPH_LIBRARIES})
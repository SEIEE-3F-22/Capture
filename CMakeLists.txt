cmake_minimum_required(VERSION 3.16)
project(Capture)

set(CMAKE_CXX_STANDARD 14)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/test)
message(Library will be generated under ${LIBRARY_OUTPUT_PATH}/test)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(pybind11)

pybind11_add_module(Capture Capture.cpp Openvino.cpp)
target_link_libraries(Capture PUBLIC ${OpenCV_LIBS})
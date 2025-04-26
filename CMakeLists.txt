cmake_minimum_required(VERSION 3.30)
project(Reconstruction)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui features2d calib3d xfeatures2d)

add_executable(Reconstruction src/main.cpp)

target_include_directories(Reconstruction PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(Reconstruction ${OpenCV_LIBS})
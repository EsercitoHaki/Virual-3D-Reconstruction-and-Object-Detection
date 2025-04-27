cmake_minimum_required(VERSION 3.16)
project(Reconstruction)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui features2d calib3d xfeatures2d)

set(LIBIGL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/libigl)

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" OFF)
option(LIBIGL_WITH_COPYLEFT "Enable libigl copyleft" ON)
option(LIBIGL_WITH_COMISO "" OFF)
option(LIBIGL_WITH_EMBREE "" OFF)
option(LIBIGL_WITH_OPENGL "" OFF)
option(LIBIGL_WITH_OPENGL_GLFW "" OFF)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "" OFF)
option(LIBIGL_WITH_PNG "" OFF)
option(LIBIGL_WITH_TETGEN "" OFF)
option(LIBIGL_WITH_TRIANGLE "" OFF)

add_subdirectory(${LIBIGL_DIR} libigl)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(ANALYSIS_DIR ${SRC_DIR}/analysis)
set(RECONSTRUCTION_DIR ${SRC_DIR}/reconstruction)
set(RENDERER_DIR ${SRC_DIR}/renderer)
set(TOOLS_DIR ${SRC_DIR}/tools)
set(UI_DIR ${SRC_DIR}/ui)
set(SFM_DIR ${SRC_DIR}/sfm)
set(SURFACE_DIR ${SRC_DIR}/surface)

set(SOURCES
    ${SRC_DIR}/main.cpp
    ${ANALYSIS_DIR}/ImageData.cpp
    ${ANALYSIS_DIR}/FeatureDetector.cpp
    ${ANALYSIS_DIR}/FeatureMatcher.cpp
    ${RENDERER_DIR}/Visualizer.cpp
    ${RECONSTRUCTION_DIR}/Exporter.cpp
    ${TOOLS_DIR}/ImageProcessor.cpp
    ${SFM_DIR}/CameraModel.cpp
    ${SFM_DIR}/PoseEstimator.cpp
    ${SFM_DIR}/Triangulator.cpp
    ${SFM_DIR}/Sfm.cpp
    ${SURFACE_DIR}/Mesh.cpp
    ${SURFACE_DIR}/IglReconstruction.cpp
)

add_executable(Reconstruction ${SOURCES})

target_include_directories(Reconstruction PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${SRC_DIR}
    ${ANALYSIS_DIR}
    ${RECONSTRUCTION_DIR}
    ${RENDERER_DIR}
    ${TOOLS_DIR}
    ${UI_DIR}
    ${SFM_DIR}
    ${SURFACE_DIR}
)

target_link_libraries(Reconstruction 
    ${OpenCV_LIBS}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    igl::core
)

target_compile_definitions(Reconstruction PRIVATE 
    -DIGL_STATIC_LIBRARY
    -DIGL_HEADER_ONLY
)
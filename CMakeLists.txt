cmake_minimum_required(VERSION 3.16)
project(Reconstruction)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui features2d calib3d xfeatures2d)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(ANALYSIS_DIR ${SRC_DIR}/analysis)
set(RECONSTRUCTION_DIR ${SRC_DIR}/reconstruction)
set(RENDERER_DIR ${SRC_DIR}/renderer)
set(TOOLS_DIR ${SRC_DIR}/tools)
set(UI_DIR ${SRC_DIR}/ui)

set(SOURCES
    ${SRC_DIR}/main.cpp
    ${ANALYSIS_DIR}/ImageData.cpp
    ${ANALYSIS_DIR}/FeatureDetector.cpp
    ${ANALYSIS_DIR}/FeatureMatcher.cpp
    ${RENDERER_DIR}/Visualizer.cpp
    ${RECONSTRUCTION_DIR}/Exporter.cpp
    ${TOOLS_DIR}/ImageProcessor.cpp
    # Các file khác như feature_detector.cpp, feature_matcher.cpp, exporter.cpp, visualizer.cpp, image_processor.cpp...
    # hiện tại chưa thấy, khi nào có thì thêm
)

add_executable(Reconstruction ${SOURCES})

target_include_directories(Reconstruction PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${SRC_DIR}
    ${ANALYSIS_DIR}
    ${RECONSTRUCTION_DIR}
    ${RENDERER_DIR}
    ${TOOLS_DIR}
    ${UI_DIR}
)

target_link_libraries(Reconstruction ${OpenCV_LIBS})

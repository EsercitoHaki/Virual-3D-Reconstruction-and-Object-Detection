#pragma once

#include "PoseEstimator.h"
#include "../analysis/FeatureMatcher.h"
#include "../analysis/ImageData.h"
#include <vector>
#include <unordered_map>
#include <opencv2/core.hpp>


namespace SFM {

struct Point3D {
    cv::Point3f position;
    cv::Vec3b color;
    std::unordered_map<size_t, size_t> observations;
};

class Triangulator {
public:
    explicit Triangulator(const CameraModel& cameraModel);

    size_t triangulateInitialPair(
        const std::vector<ImageProcessing::ImageData>& images,
        const std::pair<size_t, size_t>& initialPair,
        const std::vector<ImageProcessing::MatchData>& matches,
        const std::unordered_map<size_t, CameraPose>& cameraPoses);

    size_t triangulateNewView(
        const std::vector<ImageProcessing::ImageData>& images,
        size_t imageIdx,
        const std::vector<ImageProcessing::MatchData>& matches,
        const std::unordered_map<size_t, CameraPose>& cameraPoses,
        const std::vector<size_t>& registeredViews);

    const std::vector<Point3D>& getPoints() const;

private:
    const CameraModel& cameraModel;
    std::vector<Point3D> points3D;
};

} 

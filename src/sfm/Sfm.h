#pragma once

#include "PoseEstimator.h"
#include "Triangulator.h"
#include "../analysis/ImageData.h"
#include "../analysis/FeatureMatcher.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <unordered_map>

namespace SFM {
    class SfMReconstructor {
        public:
            SfMReconstructor(
                const std::vector<ImageProcessing::ImageData>& images,
                const std::vector<ImageProcessing::MatchData>& matches,
                const CameraModel& cameraModel
            );

            bool reconstruct();

            bool savePoses(const std::string& filename) const;
            bool savePointCloud(const std::string& filename) const;

            const Triangulator& getTriangulator() const {
                return triangulator;
            }

        private:
            const ImageProcessing::MatchData* findMatchBetweenImages(
                size_t imageIdx1, size_t imageIdx2) const;

            std::vector<ImageProcessing::ImageData> images;
            std::vector<ImageProcessing::MatchData> matches;
            PoseEstimator poseEstimator;
            Triangulator triangulator;
    };
}
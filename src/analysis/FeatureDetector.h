#pragma once

#include "ImageData.h"
#include <opencv2/features2d.hpp>
#include <memory>

namespace ImageProcessing {
    class FeatureDetector {
        public:
            enum class DetectorType {
                SIFT,
                AKAZE,
                ORB
            };

            explicit FeatureDetector(DetectorType type = DetectorType::SIFT);

            void detectFeatures(ImageData& ImageData);

            void detectFeaturesParallel(std::vector<ImageData>& images);

            static cv::Ptr<cv::Feature2D> createDetector(DetectorType type);

        
        private:
            cv::Ptr<cv::Feature2D> detector;
    };
}
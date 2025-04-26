#pragma once

#include "../analysis/ImageData.h"
#include "../analysis/FeatureDetector.h"
#include "../analysis/FeatureMatcher.h"
#include <string>
#include <vector>
#include <chrono>

namespace Tools {
    class ImageProcessor {
        public:
            explicit ImageProcessor(int maxImageSize = 1200);
            
            std::vector<ImageProcessing::ImageData> loadImagesFromDirectory(
                const std::string& imageDir, 
                int maxImages = -1);
            
            static std::vector<std::string> getSupportedImageExtensions();
            
            std::vector<ImageProcessing::MatchData> processImages(
                std::vector<ImageProcessing::ImageData>& images,
                bool sequential = true,
                ImageProcessing::FeatureDetector::DetectorType detectorType = 
                ImageProcessing::FeatureDetector::DetectorType::SIFT);
            
            double getProcessingTime() const;

        private:
            int maxImageSize;
            std::chrono::duration<double> processingTime;
    };
}
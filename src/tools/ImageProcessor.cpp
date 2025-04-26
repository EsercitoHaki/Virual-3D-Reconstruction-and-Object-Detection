#include "ImageProcessor.h"
#include <filesystem>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

namespace Tools {
    ImageProcessor::ImageProcessor(int maxImageSize) 
        : maxImageSize(maxImageSize), processingTime(0) {
    }

    std::vector<ImageProcessing::ImageData> ImageProcessor::loadImagesFromDirectory(
        const std::string& imageDir, 
        int maxImages) {
        
        std::vector<ImageProcessing::ImageData> images;
        
        auto imageExtensions = getSupportedImageExtensions();
        
        std::vector<std::string> imagePaths;
        for (const auto& entry : fs::directory_iterator(imageDir)) {
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                
                if (std::find(imageExtensions.begin(), imageExtensions.end(), extension) != imageExtensions.end()) {
                    imagePaths.push_back(entry.path().string());
                }
            }
        }
        
        std::sort(imagePaths.begin(), imagePaths.end());
        
        if (maxImages > 0 && imagePaths.size() > static_cast<size_t>(maxImages)) {
            imagePaths.resize(maxImages);
        }
        
        for (const auto& path : imagePaths) {
            ImageProcessing::ImageData imageData;
            if (imageData.loadImage(path)) {
                imageData.preprocess(maxImageSize);
                images.push_back(imageData);
            }
        }
        
        return images;
    }

    std::vector<std::string> ImageProcessor::getSupportedImageExtensions() {
        return {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
    }

    std::vector<ImageProcessing::MatchData> ImageProcessor::processImages(
        std::vector<ImageProcessing::ImageData>& images,
        bool sequential,
        ImageProcessing::FeatureDetector::DetectorType detectorType) {
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        ImageProcessing::FeatureDetector detector(detectorType);
        detector.detectFeaturesParallel(images);
        
        for (size_t i = 0; i < images.size(); i++) {
            std::cout << "Image " << i << " - Keypoints: " << images[i].getKeypointCount();
            if (!images[i].getDescriptors().empty()) {
                std::cout << " - Descriptor size: " << images[i].getDescriptors().rows 
                        << " x " << images[i].getDescriptors().cols;
            }
            std::cout << std::endl;
        }
        
        ImageProcessing::FeatureMatcher matcher;
        
        if (!images.empty() && !images[0].getDescriptors().empty()) {
            matcher.setMatcherFromDescriptorType(images[0].getDescriptors().type());
        }
        
        std::vector<ImageProcessing::MatchData> matches;
        if (sequential) {
            matches = matcher.matchSequential(images);
        } else {
            matches = matcher.matchAllPairs(images);
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        processingTime = endTime - startTime;
        
        return matches;
    }

    double ImageProcessor::getProcessingTime() const {
        return processingTime.count();
    }
}



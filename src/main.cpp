#include "analysis/ImageData.h"
#include "analysis/FeatureDetector.h"
#include "analysis/FeatureMatcher.h"
#include "renderer/Visualizer.h"
#include "reconstruction/Exporter.h"
#include "tools/ImageProcessor.h"

#include <iostream>
#include <chrono>
#include <string>

void printUsage() {
    std::cout << "Usage: ./FeatureMatching <image_directory> [max_images] [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --detector=<type>  : Feature detector type (SIFT, AKAZE, ORB), default: SIFT" << std::endl;
    std::cout << "  --match=<type>     : Matching strategy (sequential, all), default: sequential" << std::endl;
    std::cout << "  --output=<dir>     : Output directory, default: output" << std::endl;
    std::cout << "  --max-size=<pixels>: Maximum image dimension, default: 1200" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return -1;
    }
    
    std::string imageDir = argv[1];
    int maxImages = -1;
    std::string outputDir = "output";
    int maxImageSize = 1200;
    bool matchSequential = true;
    ImageProcessing::FeatureDetector::DetectorType detectorType = 
        ImageProcessing::FeatureDetector::DetectorType::SIFT;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg.find("--detector=") == 0) {
            std::string type = arg.substr(11);
            if (type == "SIFT") {
                detectorType = ImageProcessing::FeatureDetector::DetectorType::SIFT;
            } else if (type == "AKAZE") {
                detectorType = ImageProcessing::FeatureDetector::DetectorType::AKAZE;
            } else if (type == "ORB") {
                detectorType = ImageProcessing::FeatureDetector::DetectorType::ORB;
            }
        } else if (arg.find("--match=") == 0) {
            std::string type = arg.substr(8);
            if (type == "sequential") {
                matchSequential = true;
            } else if (type == "all") {
                matchSequential = false;
            }
        } else if (arg.find("--output=") == 0) {
            outputDir = arg.substr(9);
        } else if (arg.find("--max-size=") == 0) {
            maxImageSize = std::stoi(arg.substr(11));
        } else if (i == 2 && arg.find("--") != 0) {
            maxImages = std::stoi(arg);
        }
    }
    
    std::cout << "3D Reconstruction Feature Matching" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Image directory: " << imageDir << std::endl;
    std::cout << "Output directory: " << outputDir << std::endl;
    std::cout << "Maximum images: " << (maxImages == -1 ? "All" : std::to_string(maxImages)) << std::endl;
    std::cout << "Maximum image size: " << maxImageSize << std::endl;
    std::cout << "Detector: " << (detectorType == ImageProcessing::FeatureDetector::DetectorType::SIFT ? "SIFT" : 
                                 (detectorType == ImageProcessing::FeatureDetector::DetectorType::AKAZE ? "AKAZE" : "ORB")) << std::endl;
    std::cout << "Matching strategy: " << (matchSequential ? "Sequential" : "All pairs") << std::endl;
    std::cout << "--------------------------------" << std::endl;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    Tools::ImageProcessor processor(maxImageSize);
    
    std::cout << "Loading images from directory: " << imageDir << std::endl;
    auto images = processor.loadImagesFromDirectory(imageDir, maxImages);
    
    if (images.empty()) {
        std::cerr << "No valid images found in directory" << std::endl;
        return -1;
    }
    
    std::cout << "Loaded " << images.size() << " images" << std::endl;
    
    std::cout << "Processing images..." << std::endl;
    auto matches = processor.processImages(images, matchSequential, detectorType);
    
    std::cout << "Processing complete. Found " << matches.size() << " matching image pairs" << std::endl;
    
    std::cout << "Generating visualizations..." << std::endl;
    Rendering::Visualizer visualizer(outputDir);
    visualizer.visualizeAllMatches(images, matches);
    
    std::cout << "Exporting matches for reconstruction..." << std::endl;
    Reconstruction::Exporter exporter(outputDir);
    exporter.exportMatchesForReconstruction(images, matches);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Total processing time: " << elapsedTime.count() << " seconds" << std::endl;
    std::cout << "Results saved to directory: " << outputDir << std::endl;
    
    return 0;
}
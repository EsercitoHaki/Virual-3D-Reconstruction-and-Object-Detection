#include "analysis/ImageData.h"
#include "analysis/FeatureDetector.h"
#include "analysis/FeatureMatcher.h"
#include "renderer/Visualizer.h"
#include "reconstruction/Exporter.h"
#include "tools/ImageProcessor.h"
#include "sfm/Sfm.h"

#include <iostream>
#include <chrono>
#include <string>
#include <filesystem>
#include <thread>

namespace fs = std::filesystem;

void printUsage() {
    std::cout << "Usage: ./3DReconstruction <image_directory> [max_images] [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --detector=<type>  : Feature detector type (SIFT, AKAZE, ORB), default: SIFT" << std::endl;
    std::cout << "  --match=<type>     : Matching strategy (sequential, all), default: sequential" << std::endl;
    std::cout << "  --output=<dir>     : Output directory, default: output" << std::endl;
    std::cout << "  --max-size=<pixels>: Maximum image dimension, default: 1200" << std::endl;
    std::cout << "  --neighbors=<count>: Number of neighbor images to match in sequential mode, default: 2" << std::endl;
    std::cout << "  --min-matches=<n>  : Minimum number of matches to consider a valid pair, default: 20" << std::endl;
    std::cout << "  --threads=<n>      : Number of threads to use (0=auto), default: 0" << std::endl;
    std::cout << "  --no-vis           : Skip visualization generation" << std::endl;
    std::cout << "  --help             : Show this help message" << std::endl;
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
    int neighborCount = 2;
    int minMatches = 20;
    int numThreads = 4;
    bool generateVisualization = true;
    
    ImageProcessing::FeatureDetector::DetectorType detectorType = 
        ImageProcessing::FeatureDetector::DetectorType::SIFT;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage();
            return 0;
        } else if (arg.find("--detector=") == 0) {
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
            try {
                maxImageSize = std::stoi(arg.substr(11));
            } catch (const std::invalid_argument&) {
                std::cerr << "Error: Invalid max size value: " << arg.substr(11) << std::endl;
                return -1;
            }
        } else if (arg.find("--neighbors=") == 0) {
            try {
                neighborCount = std::stoi(arg.substr(12));
            } catch (const std::invalid_argument&) {
                std::cerr << "Error: Invalid neighbor count: " << arg.substr(12) << std::endl;
                return -1;
            }
        } else if (arg.find("--min-matches=") == 0) {
            try {
                minMatches = std::stoi(arg.substr(14));
            } catch (const std::invalid_argument&) {
                std::cerr << "Error: Invalid minimum matches: " << arg.substr(14) << std::endl;
                return -1;
            } 
        } else if (arg.find("--threads=") == 0) {
            try {
                numThreads = std::stoi(arg.substr(10));
            } catch (const std::invalid_argument&) {
                std::cerr << "Error: Invalid thread count: " << arg.substr(10) << std::endl;
                return -1;
            }
        } else if (arg == "--no-vis") {
            generateVisualization = false;
        } else if (i == 2 && arg.find("--") != 0) {
            try {
                maxImages = std::stoi(arg);
            } catch (const std::invalid_argument&) {
                std::cerr << "Error: Invalid maximum images value: " << arg << std::endl;
                return -1;
            }
        }
    }
    
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Print configuration
    std::cout << "3D Reconstruction" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Image directory: " << imageDir << std::endl;
    std::cout << "Output directory: " << outputDir << std::endl;
    std::cout << "Maximum number of images: " << (maxImages == -1 ? "All" : std::to_string(maxImages)) << std::endl;
    std::cout << "Maximum image size: " << maxImageSize << std::endl;
    std::cout << "Neighbor count: " << neighborCount << std::endl;
    std::cout << "Minimum matches: " << minMatches << std::endl;
    std::cout << "Thread count: " << numThreads << std::endl;
    std::cout << "Feature detector: " 
              << (detectorType == ImageProcessing::FeatureDetector::DetectorType::SIFT ? "SIFT" : 
                  (detectorType == ImageProcessing::FeatureDetector::DetectorType::AKAZE ? "AKAZE" : "ORB")) 
              << std::endl;
    std::cout << "Matching strategy: " << (matchSequential ? "Sequential" : "All pairs") << std::endl;
    std::cout << "Visualization: " << (generateVisualization ? "Enabled" : "Disabled") << std::endl;
    std::cout << "--------------------------------" << std::endl;
    
    // Image processing
    Tools::ImageProcessor processor(maxImageSize);
    
    std::cout << "Loading images from: " << imageDir << std::endl;
    auto images = processor.loadImagesFromDirectory(imageDir, maxImages);
    
    if (images.empty()) {
        std::cerr << "Error: No valid images found in the directory" << std::endl;
        return -1;
    }
    
    std::cout << "Loaded " << images.size() << " images" << std::endl;
    
    std::cout << "Processing images..." << std::endl;
    auto matches = processor.processImages(images, matchSequential, detectorType);
    
    std::cout << "Processing completed. Found " << matches.size() << " matching image pairs" << std::endl;
    
    // Visualization
    if (generateVisualization) {
        std::cout << "Generating visualizations..." << std::endl;
        Rendering::Visualizer visualizer(outputDir);
        visualizer.visualizeAllMatches(images, matches, minMatches);
    }
    
    // Reconstruction
    std::cout << "Starting 3D Reconstruction..." << std::endl;
    SFM::CameraModel cameraModel = SFM::CameraModel::estimateFromImageSize(
        images[0].getImage().cols, 
        images[0].getImage().rows
    );
    
    SFM::SfMReconstructor reconstructor(images, matches, cameraModel);
    
    if (!reconstructor.reconstruct()) {
        std::cerr << "3D Reconstruction failed" << std::endl;
        return -1;
    }
    
    std::cout << "3D Reconstruction completed successfully" << std::endl;
    
    // Export reconstruction data
    std::cout << "Exporting reconstruction data..." << std::endl;
    Reconstruction::Exporter exporter(outputDir);
    exporter.exportMatchesForReconstruction(images, matches, minMatches);
    
    // End timing and report
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Total processing time: " << elapsedTime.count() << " seconds" << std::endl;
    std::cout << "Results saved to: " << outputDir << std::endl;
    
    return 0;
}
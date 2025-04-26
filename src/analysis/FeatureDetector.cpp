#include "FeatureDetector.h"

#include <thread>
#include <iostream>

namespace ImageProcessing {
    FeatureDetector::FeatureDetector(DetectorType type) {
        detector = createDetector(type);
    }

    void FeatureDetector::detectFeatures(ImageData& imageData) {
        detector->detectAndCompute(imageData.grayImage, cv::noArray(),
                                    imageData.keypoints, imageData.descriptors);
    }

    void FeatureDetector::detectFeaturesParallel(std::vector<ImageData>& images) {
        std::vector<std::thread> threads;
        const int numThreads = std::min(static_cast<int>(images.size()),
                                        static_cast<int>(std::thread::hardware_concurrency()));

        std::cout << "Using " << numThreads << " threads for feature detection" << std::endl;

        std::vector<std::vector<ImageData*>> threadWorkload(numThreads);

        for (size_t i = 0; i < images.size(); i++) {
            threadWorkload[i % numThreads].push_back(&images[i]);
        }

        for (int t = 0; t < images.size(); t++) {
            threads.emplace_back([this, &threadWorkload, t]() {
                for (auto* imageData : threadWorkload[t]) {
                    this->detectFeatures(*imageData);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }

    cv::Ptr<cv::Feature2D> FeatureDetector::createDetector(DetectorType type) {
        switch (type) {
            case DetectorType::SIFT:
                return cv::SIFT::create(
                    0,      // nfeatures (0 = no limit)
                    3,      // nOctaveLayers
                    0.04,   // contrastThreshold
                    10,     // edgeThreshold
                    1.6     // sigma
                );
            
            case DetectorType::AKAZE:
                return cv::AKAZE::create(
                    cv::AKAZE::DESCRIPTOR_MLDB,   // descriptor_type
                    0,                           // descriptor_size
                    0,                           // descriptor_channels
                    0.001f,                      // threshold
                    4,                           // octaves
                    4,                           // octave_layers
                    cv::KAZE::DIFF_PM_G2         // diffusivity
                );
                
            case DetectorType::ORB:
                return cv::ORB::create(
                    2000,                      // nfeatures
                    1.2f,                      // scaleFactor
                    8,                         // nlevels
                    31,                        // edgeThreshold
                    0,                         // firstLevel
                    2,                         // WTA_K
                    cv::ORB::HARRIS_SCORE,     // scoreType
                    31,                        // patchSize
                    20                         // fastThreshold
                );
                
            default:
                return cv::SIFT::create();
        }
    }
}
#include "FeatureDetector.h"

#include <thread>
#include <future>
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
        unsigned int numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 4;

        numThreads = std::min(numThreads, 8u);
        numThreads = std::min(numThreads, static_cast<unsigned int>(images.size()));

        std::cout << "Using " << numThreads << " threads for feature detection" << std::endl;

        std::vector<std::future<void>> futures;

        for (size_t t = 0; t < numThreads; ++t) {
            futures.push_back(std::async(std::launch::async, [&, t]() {
                for (size_t i = t; i < images.size(); i += numThreads) {
                    detectFeatures(images[i]);
                    std::cout << "Detected " << images[i].keypoints.size()
                              << " features in image " << (i + 1) << "/" << images.size()
                              << ": " << images[i].getPath() << std::endl;
                }
            }));
        }

        for (auto& f : futures) {
            f.wait();
        }
    }

    cv::Ptr<cv::Feature2D> FeatureDetector::createDetector(DetectorType type) {
        switch (type) {
            case DetectorType::SIFT:
                return cv::SIFT::create();
            case DetectorType::AKAZE:
                return cv::AKAZE::create();
            case DetectorType::ORB:
                return cv::ORB::create(2000);
            default:
                return cv::SIFT::create();
        }
    }
}
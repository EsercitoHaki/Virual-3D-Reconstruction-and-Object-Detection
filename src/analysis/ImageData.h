#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace ImageProcessing {
    class ImageData {
        public:
            ImageData() = default;

            explicit ImageData(const std::string& path);

            bool loadImage(const std::string& path);

            void preprocess(const int maxImageSize);

            const std::string& getPath() const { return path; }

            const cv::Mat& getImage() const { return image; }

            const cv::Mat& getGrayImage() const { return grayImage; }

            const std::vector<cv::KeyPoint>& getKeypoints() const { return keypoints; }

            const cv::Mat& getDescriptors() const { return descriptors; }

            size_t getKeypointCount() const { return keypoints.size(); }

            std::pair<int, int> getDimensions() const;

            friend class FeatureDetector;

        private:
            cv::Mat image;
            cv::Mat grayImage;
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            std::string path;
    };
}
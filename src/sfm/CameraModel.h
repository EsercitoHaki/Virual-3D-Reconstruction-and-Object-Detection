#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace SFM {
    class CameraModel {
        public:
            CameraModel();

            CameraModel(double focalLength, 
                const cv::Point2d& principalPoint,
                const cv::Size& imageSize,
                const cv::Mat& distortionCoeffs = cv::Mat::zeros(5, 1, CV_64F)
            );

            static CameraModel estimateFromImageSize(int imageWidth, int imageHeight);

            cv::Mat getCameraMatrix() const;

            cv::Mat getDistortionCoeffs() const;

            double getFocalLength() const;

            cv::Point2d getPrincipalPoint() const;

            cv::Size getImageSize() const;

            bool saveToFile(const std::string& filename) const;

            bool loadFromFile(const std::string& filename);

        private:
            double fx;
            double fy;
            cv::Point2d principalPoint;
            cv::Size imageSize;
            cv::Mat distCoeffs;
        
    };
}
#include "CameraModel.h"

#include <fstream>
#include <iostream>

namespace SFM {
    CameraModel::CameraModel() 
            : fx(0), fy(0), principalPoint(0, 0), imageSize(0, 0) {
        distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    }

    CameraModel::CameraModel(double focalLength, 
        const cv::Point2d& principalPoint,
        const cv::Size& imageSize,
        const cv::Mat& distortionCoeffs)
            : fx(focalLength), fy(focalLength), 
            principalPoint(principalPoint), 
            imageSize(imageSize) {

        if (distortionCoeffs.empty()) {
            distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
        } else {
            distortionCoeffs.copyTo(distCoeffs);
        }
    }

    CameraModel CameraModel::estimateFromImageSize(int imageWidth, int imageHeight) {
        double focalLength = 1.2 * std::max(imageWidth, imageHeight);

        cv::Point2d pp(imageWidth / 2.0, imageHeight / 2.0);

        cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);

        return CameraModel(focalLength, pp, cv::Size(imageWidth, imageHeight), distCoeffs);
    }

    cv::Mat CameraModel::getCameraMatrix() const {
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0, 0) = fx;
        K.at<double>(1, 1) = fy;
        K.at<double>(0, 2) = principalPoint.x;
        K.at<double>(1, 2) = principalPoint.y;
        return K;
    }

    cv::Mat CameraModel::getDistortionCoeffs() const {
        return distCoeffs;
    }

    double CameraModel::getFocalLength() const {
        return fx;
    }

    cv::Point2d CameraModel::getPrincipalPoint() const {
        return principalPoint;
    }

    cv::Size CameraModel::getImageSize() const {
        return imageSize;
    }

    bool CameraModel::saveToFile(const std::string& filename) const {
        std::ofstream file(filename);

        if(!file.is_open()) {
            return false;
        }

        file << "fx: " << fx << std::endl;
        file << "fy: " << fy << std::endl;
        file << "cx: " << principalPoint.x << std::endl;
        file << "cy: " << principalPoint.y << std::endl;
        file << "width: " << imageSize.width << std::endl;
        file << "height: " << imageSize.height << std::endl;

        file << "distortion: ";
        for (int i = 0; i < distCoeffs.rows; i++) {
            file << distCoeffs.at<double>(i, 0) << " ";
        }
        file << std::endl;
        
        file.close();
        return true;
    }

    bool CameraModel::loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        std::string line, key;
        double value;
        
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            
            if (std::getline(iss, key, ':')) {
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                
                if (key == "fx") {
                    iss >> fx;
                } else if (key == "fy") {
                    iss >> fy;
                } else if (key == "cx") {
                    iss >> principalPoint.x;
                } else if (key == "cy") {
                    iss >> principalPoint.y;
                } else if (key == "width") {
                    iss >> imageSize.width;
                } else if (key == "height") {
                    iss >> imageSize.height;
                } else if (key == "distortion") {
                    distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
                    for (int i = 0; i < 5 && !iss.eof(); i++) {
                        iss >> distCoeffs.at<double>(i, 0);
                    }
                }
            }
        }
        
        file.close();
        return true;
    }
}
#include "ImageData.h"
#include <iostream>

namespace ImageProcessing {
    ImageData::ImageData(const std::string& path) {
        loadImage(path);
    }

    bool ImageData::loadImage(const std::string& path) {
        this->path = path;
        image = cv::imread(path, cv::IMREAD_COLOR);

        if (image.empty()) {
            std::cerr << "Could not read image: " << path << std::endl;
            return false;
        }

        std::cout << "Loaded image: " << path << " ("
                  << image.cols << "x" << image.rows << ")" << std::endl;

        return true;
    }

    void ImageData::preprocess(const int maxImageSize) {
        if (image.cols > maxImageSize || image.rows > maxImageSize) {
            double scale = std::min(
                double(maxImageSize) / image.cols,
                double(maxImageSize) / image.rows
            );
            cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_AREA);
        }

        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

        cv::equalizeHist(grayImage, grayImage);

        cv::GaussianBlur(grayImage, grayImage, cv::Size(3, 3), 0);
    }

    std::pair<int, int> ImageData::getDimensions() const {
        return std::make_pair(image.cols, image.rows);
    }
}
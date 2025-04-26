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

        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        return true;
    }

    void ImageData::preprocess(const int maxImageSize) {
        if (maxImageSize > 0) {
            double scale = 1.0;
            if (image.cols > maxImageSize || image.rows > maxImageSize) {
                if (image.cols > image.rows) {
                    scale = static_cast<double>(maxImageSize) / image.cols;
                }
                else {
                    scale = static_cast<double>(maxImageSize) / image.rows;
                }

                cv::resize(image, image, cv::Size(static_cast<int>(image.cols * scale),
                                                    static_cast<int>(image.rows * scale)));

                cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
            }
        }

        cv::Mat blurred;
        cv::GaussianBlur(grayImage, blurred, cv::Size(3, 3), 0);

        cv::equalizeHist(blurred, grayImage);
    }

    std::pair<int, int> ImageData::getDimensions() const {
        return {image.cols, image.rows};
    }
}
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        return -1;
    }

    std::string imagePath = argv[1];
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cout << "Unable to read image file: " << imagePath << std::endl;
        return -1;
    }

    std::cout << "Number of color channels: " << image.channels() << std::endl;

    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image", image);

    while (true) {
        cv::waitKey(1);

        if (cv::getWindowProperty("Image", cv::WND_PROP_VISIBLE) < 1) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <chrono>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: ./FeatureMatching <image1_path> <image2_path>" << std::endl;
        return -1;
    }

    std::string imagePath1 = argv[1];
    std::string imagePath2 = argv[2];
    
    cv::Mat image1 = cv::imread(imagePath1, cv::IMREAD_COLOR);
    cv::Mat image2 = cv::imread(imagePath2, cv::IMREAD_COLOR);

    if (image1.empty() || image2.empty()) {
        std::cout << "Unable to read one or both image files" << std::endl;
        return -1;
    }

    // Resize images for faster processing if they are too large
    const int MAX_IMAGE_SIZE = 1200; // Maximum dimension (width or height)
    if (image1.cols > MAX_IMAGE_SIZE || image1.rows > MAX_IMAGE_SIZE) {
        double scale = std::min(double(MAX_IMAGE_SIZE) / image1.cols, double(MAX_IMAGE_SIZE) / image1.rows);
        cv::resize(image1, image1, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    if (image2.cols > MAX_IMAGE_SIZE || image2.rows > MAX_IMAGE_SIZE) {
        double scale = std::min(double(MAX_IMAGE_SIZE) / image2.cols, double(MAX_IMAGE_SIZE) / image2.rows);
        cv::resize(image2, image2, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    cv::Mat grayImage1, grayImage2;
    cv::cvtColor(image1, grayImage1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, grayImage2, cv::COLOR_BGR2GRAY);

    // Optional: Apply contrast enhancement for better feature detection
    cv::equalizeHist(grayImage1, grayImage1);
    cv::equalizeHist(grayImage2, grayImage2);

    // Timer to measure performance
    auto start = std::chrono::high_resolution_clock::now();

    // For 3D reconstruction, SIFT or AKAZE typically perform better than ORB
    // Uncomment the feature detector you want to use:
    
    // Option 1: ORB (fastest but less accurate for 3D reconstruction)
    // cv::Ptr<cv::Feature2D> detector = cv::ORB::create(
    //     2000,                // nfeatures
    //     1.2f,                // scaleFactor
    //     8,                   // nlevels
    //     31,                  // edgeThreshold
    //     0,                   // firstLevel
    //     2,                   // WTA_K
    //     cv::ORB::HARRIS_SCORE, // scoreType
    //     31,                  // patchSize
    //     20                   // fastThreshold
    // );
    
    // Option 2: SIFT (better for 3D reconstruction but slower)
    cv::Ptr<cv::Feature2D> detector = cv::SIFT::create(
        0,      // nfeatures (0 = no limit)
        3,      // nOctaveLayers
        0.04,   // contrastThreshold
        10,     // edgeThreshold
        1.6     // sigma
    );
    
    // Option 3: AKAZE (good balance between speed and accuracy)
    // cv::Ptr<cv::Feature2D> detector = cv::AKAZE::create(
    //     cv::AKAZE::DESCRIPTOR_MLDB,  // descriptor_type
    //     0,                          // descriptor_size
    //     0,                          // descriptor_channels
    //     0.001f,                     // threshold
    //     4,                          // octaves
    //     4,                          // octave_layers
    //     cv::KAZE::DIFF_PM_G2        // diffusivity
    // );

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            detector->detectAndCompute(grayImage1, cv::noArray(), keypoints1, descriptors1);
        }
        #pragma omp section
        {
            detector->detectAndCompute(grayImage2, cv::noArray(), keypoints2, descriptors2);
        }
    }
    
    std::cout << "Image 1 - So luong keypoints: " << keypoints1.size() << std::endl;
    std::cout << "Image 1 - Kich thuoc descriptor: " << descriptors1.rows << " x " << descriptors1.cols << std::endl;
    
    std::cout << "Image 2 - So luong keypoints: " << keypoints2.size() << std::endl;
    std::cout << "Image 2 - Kich thuoc descriptor: " << descriptors2.rows << " x " << descriptors2.cols << std::endl;

    cv::Ptr<cv::DescriptorMatcher> matcher;
    
    if (descriptors1.type() == CV_32F) {
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    } 
    else {
        matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    }

    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);
    
    const float ratioThresh = 0.75f;
    std::vector<cv::DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i].size() >= 2 && 
            knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }
    
    std::sort(goodMatches.begin(), goodMatches.end(), 
              [](const cv::DMatch& a, const cv::DMatch& b) {
                  return a.distance < b.distance;
              });

    const int numBestMatches = std::min(100, static_cast<int>(goodMatches.size()));
    if (goodMatches.size() > numBestMatches) {
        goodMatches.resize(numBestMatches);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "So luong matches tot: " << goodMatches.size() << std::endl;
    std::cout << "Thoi gian xu ly: " << elapsed.count() << " seconds" << std::endl;
    
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : goodMatches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }
    
    cv::Mat fundamentalMatrix;
    if (points1.size() >= 8) {
        fundamentalMatrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3.0, 0.99);
        std::cout << "Da tim thay fundamental matrix" << std::endl;
    }
    
    cv::Mat imgMatches;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, imgMatches,
                   cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                   std::vector<char>(), cv::DrawMatchesFlags::DEFAULT);
    
    cv::Mat keypointsImage1, keypointsImage2;
    cv::drawKeypoints(image1, keypoints1, keypointsImage1, cv::Scalar(0, 0, 255), 
                     cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(image2, keypoints2, keypointsImage2, cv::Scalar(0, 0, 255), 
                     cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    std::string matchText = "Matches: " + std::to_string(goodMatches.size()) + 
                           " / " + std::to_string(std::min(keypoints1.size(), keypoints2.size()));
    cv::putText(imgMatches, matchText, cv::Point(20, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    
    cv::namedWindow("Image 1 Keypoints", cv::WINDOW_NORMAL);
    cv::namedWindow("Image 2 Keypoints", cv::WINDOW_NORMAL);
    cv::namedWindow("Feature Matches", cv::WINDOW_NORMAL);
    
    cv::imshow("Image 1 Keypoints", keypointsImage1);
    cv::imshow("Image 2 Keypoints", keypointsImage2);
    cv::imshow("Feature Matches", imgMatches);
    
    // Save results
    cv::imwrite("keypoints1.jpg", keypointsImage1);
    cv::imwrite("keypoints2.jpg", keypointsImage2);
    cv::imwrite("matches.jpg", imgMatches);

    while (true) {
        int key = cv::waitKey(1);
        if (key == 27) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
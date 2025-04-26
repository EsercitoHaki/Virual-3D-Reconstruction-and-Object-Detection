#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <chrono>
#include <vector>
#include <filesystem>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <fstream>

namespace fs = std::filesystem;

struct ImageData {
    cv::Mat image;
    cv::Mat grayImage;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::string path;
};

struct MatchData
{
    int imageIdx1;
    int imageIdx2;
    std::vector<cv::DMatch> matches;
    cv::Mat fundamentalMatrix;
};

void detectFeatures(ImageData& imageData, cv::Ptr<cv::Feature2D>& detector) {
    detector->detectAndCompute(imageData.grayImage, cv::noArray(),
                                imageData.keypoints, imageData.descriptors);
}

MatchData matchFeatures(const ImageData& image1, const ImageData& image2,
                        int idx1, int idx2, cv::Ptr<cv::DescriptorMatcher>& matcher) {
    MatchData result;
    result.imageIdx1 = idx1;
    result.imageIdx2 = idx2;

    if (image1.descriptors.empty() || image2.descriptors.empty()) {
        return result;
    }

    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(image1.descriptors, image2.descriptors, knnMatches, 2);

    const float ratioThresh = 0.75f;
    std::vector<cv::DMatch> goodMatches;
    for (const auto& match : knnMatches) {
        if (match.size() >= 2 && match[0].distance < ratioThresh * match[1].distance) {
            goodMatches.push_back(match[0]);
        }
    }

    std::sort(goodMatches.begin(), goodMatches.end(),
            [](const cv::DMatch& a, const cv::DMatch& b) {
                return a.distance < b.distance;
            });

    const int maxMatches = 500;

    if (goodMatches.size() > maxMatches) {
        goodMatches.resize(maxMatches);
    }

    result.matches = goodMatches;

    if (goodMatches.size() >= 8) {
        std::vector<cv::Point2f> points1, points2;
        for (const auto& match : goodMatches) {
            points1.push_back(image1.keypoints[match.queryIdx].pt);
            points2.push_back(image2.keypoints[match.trainIdx].pt);
        }

        result.fundamentalMatrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3.0, 0.99);
    }

    return result;
}

void preprocessImage(ImageData& imageData, const int MAX_IMAGE_SIZE) {
    if (imageData.image.cols > MAX_IMAGE_SIZE || imageData.image.rows > MAX_IMAGE_SIZE) {
        double scale = std::min(
            double(MAX_IMAGE_SIZE) / imageData.image.cols,
            double(MAX_IMAGE_SIZE) / imageData.image.rows
        );
        cv::resize(imageData.image, imageData.image, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    cv::cvtColor(imageData.image, imageData.grayImage, cv::COLOR_BGR2GRAY);

    cv::equalizeHist(imageData.grayImage, imageData.grayImage);

    cv::GaussianBlur(imageData.grayImage, imageData.grayImage, cv::Size(3, 3), 0);
}

void visualizeMatches(const std::vector<ImageData>& images,
                      const std::vector<MatchData>& allMatches,
                      const std::string& outputDir) {
    if (!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
    }

    for (size_t i = 0; i < images.size(); i++) {
        cv::Mat keypointsImage;
        cv::drawKeypoints(images[i].image, images[i].keypoints, keypointsImage,
                        cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        std::string filename = outputDir + "/keypoints_" + std::to_string(i) + ".jpg";
        cv::imwrite(filename, keypointsImage);
    }

    for (const auto& match : allMatches) {
        if (match.matches.size() >= 20) {
            cv::Mat imgMatches;
            cv::drawMatches(
                images[match.imageIdx1].image, images[match.imageIdx1].keypoints,
                images[match.imageIdx2].image, images[match.imageIdx2].keypoints,
                match.matches, imgMatches,
                cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                std::vector<char>(), cv::DrawMatchesFlags::DEFAULT
            );

            std::string matchText = "Matches: " + std::to_string(match.matches.size()) +
                                    " between images " + std::to_string(match.imageIdx2) +
                                    " and " + std::to_string(match.imageIdx2);
            
            cv::putText(imgMatches, matchText, cv::Point(20, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

            std::string filename = outputDir + "/matches_" + 
                                    std::to_string(match.imageIdx1) + "_" +
                                    std::to_string(match.imageIdx2) + ".jpg";

            cv::imwrite(filename, imgMatches);
        }
    }

    const int graphSize = 800;

    cv::Mat matchGraph(graphSize, graphSize, CV_8UC3, cv::Scalar(255, 255, 255));

    const int numImages = images.size();
    std::vector<cv::Point> nodePositions;

    for (int i = 0; i < numImages; i++) {
        float angle = 2 * CV_PI * i / numImages;
        int radius = graphSize / 3;
        cv::Point pos (
            graphSize / 2 + radius * cos(angle),
            graphSize / 2 + radius * sin(angle)
        );

        nodePositions.push_back(pos);

        cv::circle(matchGraph, pos, 20, cv::Scalar(0, 0, 0), -1);
        cv::putText(matchGraph, std::to_string(i),
                    cv::Point(pos.x - 5, pos.y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    for (const auto& match : allMatches) {
        if (match.matches.size() >= 20) {
            cv::Point p1 = nodePositions[match.imageIdx1];
            cv::Point p2 = nodePositions[match.imageIdx2];
            
            int thickness = std::min(8, static_cast<int>(1 + match.matches.size() / 50));
            cv::line(matchGraph, p1, p2, cv::Scalar(0, 0, 255), thickness);
            
            cv::Point labelPos((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
            cv::putText(matchGraph, std::to_string(match.matches.size()),
                       labelPos, cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                       cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }
    }
    
    cv::imwrite(outputDir + "/match_graph.jpg", matchGraph);
}

void exportMatchesForReconstruction(const std::vector<ImageData>& images,
                                    const std::vector<MatchData>& allMatches,
                                    const std::string& outputDir) {
    std::string outputFile = outputDir + "/matches.txt";
    std::ofstream output(outputFile);

    if (!output.is_open()) {
        std::cerr << "Failed to open output file for matches" << std::endl;
        return;
    }

    output << "# Image paths:" << std::endl;
    for (size_t i = 0; i < images.size(); i++) {
        output << i << " " << images[i].path << std::endl;
    }

    output << "# Matches (image1_idx image2_idx num_matches):" << std::endl;

    for (const auto& match : allMatches) {
        if (match.matches.size() >= 20) {
            output << match.imageIdx1 << " " << match.imageIdx2 << " " 
            << match.matches.size() << std::endl;

            for (const auto& m : match.matches) {
                const cv::KeyPoint& kp1 = images[match.imageIdx1].keypoints[m.queryIdx];
                const cv::KeyPoint& kp2 = images[match.imageIdx2].keypoints[m.trainIdx];

                output << kp1.pt.x << " " << kp1.pt.y << " " 
                << kp2.pt.x << " " << kp2.pt.y << " " 
                << m.distance << std::endl;
            }
        }
    }

    output.close();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./FeatureMatching <image_directory> [max_images]" << std::endl;
        return -1;
    }
    
    std::string imageDir = argv[1];
    int maxImages = (argc >= 3) ? std::stoi(argv[2]) : -1;
    
    const int MAX_IMAGE_SIZE = 1200;        
    const std::string outputDir = "output"; 
    
    std::vector<ImageData> images;
    
    std::cout << "Loading images from: " << imageDir << std::endl;
    
    std::vector<std::string> imageExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
    
    std::vector<std::string> imagePaths;
    for (const auto& entry : fs::directory_iterator(imageDir)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            if (std::find(imageExtensions.begin(), imageExtensions.end(), extension) != imageExtensions.end()) {
                imagePaths.push_back(entry.path().string());
            }
        }
    }
    
    std::sort(imagePaths.begin(), imagePaths.end());
    
    if (maxImages > 0 && imagePaths.size() > static_cast<size_t>(maxImages)) {
        imagePaths.resize(maxImages);
    }
    
    for (const auto& path : imagePaths) {
        ImageData imageData;
        imageData.path = path;
        imageData.image = cv::imread(path, cv::IMREAD_COLOR);
        
        if (imageData.image.empty()) {
            std::cerr << "Could not read image: " << path << std::endl;
            continue;
        }
        
        preprocessImage(imageData, MAX_IMAGE_SIZE);
        images.push_back(imageData);
        
        std::cout << "Loaded image: " << path << " (" 
                 << imageData.image.cols << "x" << imageData.image.rows << ")" << std::endl;
    }
    
    if (images.empty()) {
        std::cerr << "No valid images found in directory" << std::endl;
        return -1;
    }
    
    std::cout << "Loaded " << images.size() << " images" << std::endl;
    
    // Option 1: SIFT (better for 3D reconstruction but slower)
    cv::Ptr<cv::Feature2D> detector = cv::SIFT::create(
        0,      // nfeatures (0 = no limit)
        3,      // nOctaveLayers
        0.04,   // contrastThreshold
        10,     // edgeThreshold
        1.6     // sigma
    );
    
    // Option 2: AKAZE (good balance between speed and accuracy)
    /*
    cv::Ptr<cv::Feature2D> detector = cv::AKAZE::create(
        cv::AKAZE::DESCRIPTOR_MLDB,   // descriptor_type
        0,                           // descriptor_size
        0,                           // descriptor_channels
        0.001f,                      // threshold
        4,                           // octaves
        4,                           // octave_layers
        cv::KAZE::DIFF_PM_G2         // diffusivity
    );
    */
    
    // Option 3: ORB (fastest but less accurate for 3D reconstruction)
    /*
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create(
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
    */
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    {
        std::vector<std::thread> threads;
        const int numThreads = std::min(static_cast<int>(images.size()), 
                                      static_cast<int>(std::thread::hardware_concurrency()));
        std::cout << "Using " << numThreads << " threads for feature detection" << std::endl;
        
        std::vector<std::vector<ImageData*>> threadWorkload(numThreads);
        
        for (size_t i = 0; i < images.size(); i++) {
            threadWorkload[i % numThreads].push_back(&images[i]);
        }
        
        for (int t = 0; t < numThreads; t++) {
            threads.emplace_back([&detector, &threadWorkload, t]() {
                for (auto* imageData : threadWorkload[t]) {
                    detectFeatures(*imageData, detector);
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    for (size_t i = 0; i < images.size(); i++) {
        std::cout << "Image " << i << " - Keypoints: " << images[i].keypoints.size();
        if (!images[i].descriptors.empty()) {
            std::cout << " - Descriptor size: " << images[i].descriptors.rows 
                     << " x " << images[i].descriptors.cols;
        }
        std::cout << std::endl;
    }
    
    cv::Ptr<cv::DescriptorMatcher> matcher;
    if (!images.empty() && !images[0].descriptors.empty() && images[0].descriptors.type() == CV_32F) {
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    } else {
        matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    }
    
    std::vector<MatchData> allMatches;
    
    bool matchSequential = true;
    
    if (matchSequential) {
        for (size_t i = 0; i < images.size() - 1; i++) {
            MatchData match = matchFeatures(images[i], images[i+1], i, i+1, matcher);
            if (!match.matches.empty()) {
                allMatches.push_back(match);
                std::cout << "Matched image " << i << " with " << (i+1) 
                         << ": " << match.matches.size() << " matches" << std::endl;
            }
            
            if (i + 2 < images.size()) {
                MatchData match2 = matchFeatures(images[i], images[i+2], i, i+2, matcher);
                if (!match2.matches.empty()) {
                    allMatches.push_back(match2);
                    std::cout << "Matched image " << i << " with " << (i+2) 
                             << ": " << match2.matches.size() << " matches" << std::endl;
                }
            }
        }
    } else {
        for (size_t i = 0; i < images.size(); i++) {
            for (size_t j = i + 1; j < images.size(); j++) {
                MatchData match = matchFeatures(images[i], images[j], i, j, matcher);
                if (!match.matches.empty()) {
                    allMatches.push_back(match);
                    std::cout << "Matched image " << i << " with " << j 
                             << ": " << match.matches.size() << " matches" << std::endl;
                }
            }
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    
    std::cout << "Total processing time: " << elapsedTime.count() << " seconds" << std::endl;
    std::cout << "Total number of image pairs with matches: " << allMatches.size() << std::endl;
    
    visualizeMatches(images, allMatches, outputDir);
    
    exportMatchesForReconstruction(images, allMatches, outputDir);
    
    std::cout << "Results saved to directory: " << outputDir << std::endl;
    
    return 0;
}
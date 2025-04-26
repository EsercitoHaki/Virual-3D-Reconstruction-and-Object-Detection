#include "FeatureMatcher.h"

#include <iostream>
#include <algorithm>

namespace ImageProcessing {
    FeatureMatcher::FeatureMatcher() {
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }
    
    void FeatureMatcher::setMatcherFromDescriptorType(int descriptorType) {
        if (descriptorType == CV_8U) {
            matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
        } else {
            matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        }
        currentDescriptorType = descriptorType;
    }

    cv::Mat FeatureMatcher::computeFundamentalMatrix(
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const std::vector<cv::DMatch>& matches,
        std::vector<cv::DMatch>& inlierMatches) {
        
        if (matches.size() < 8) {
            // Need at least 8 points to compute fundamental matrix
            inlierMatches = matches;
            return cv::Mat();
        }
        
        // Extract matching points
        std::vector<cv::Point2f> points1, points2;
        for (const auto& match : matches) {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }
        
        // Use RANSAC to compute fundamental matrix and find inliers
        std::vector<uchar> inliersMask;
        cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3.0, 0.99, inliersMask);
        
        // Keep only inlier matches
        inlierMatches.clear();
        for (size_t i = 0; i < inliersMask.size(); i++) {
            if (inliersMask[i]) {
                inlierMatches.push_back(matches[i]);
            }
        }
        
        std::cout << "  Found " << inlierMatches.size() << " inliers out of " 
                  << matches.size() << " matches." << std::endl;
                  
        return F;
    }
    
    MatchData FeatureMatcher::matchFeatures(const ImageData& image1, const ImageData& image2, int idx1, int idx2) {
        MatchData result;
        result.imageIdx1 = idx1;
        result.imageIdx2 = idx2;
    
        if (image1.getDescriptors().empty() || image2.getDescriptors().empty()) {
            return result;
        }

        if (currentDescriptorType != image1.getDescriptors().type()) {
            setMatcherFromDescriptorType(image1.getDescriptors().type());
        }

        std::vector<std::vector<cv::DMatch>> knnMatches;
        try {
            matcher->knnMatch(image1.getDescriptors(), image2.getDescriptors(), knnMatches, 2);
        } catch (const cv::Exception& e) {
            std::cerr << "Error during matching: " << e.what() << std::endl;
            return result;
        }
    
        std::vector<cv::DMatch> goodMatches;
        for (const auto& match : knnMatches) {
            if (match.size() >= 2 && match[0].distance < ratioTestThreshold * match[1].distance) {
                goodMatches.push_back(match[0]);
            }
        }
    
        std::sort(goodMatches.begin(), goodMatches.end());
        if (goodMatches.size() > static_cast<size_t>(maxMatches)) {
            goodMatches.resize(maxMatches);
        }
    
        if (goodMatches.size() >= 8) {
            std::vector<cv::Point2f> points1, points2;
            for (const auto& match : goodMatches) {
                points1.push_back(image1.getKeypoints()[match.queryIdx].pt);
                points2.push_back(image2.getKeypoints()[match.trainIdx].pt);
            }

            std::vector<uchar> inliersMask;
            result.fundamentalMatrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3.0, 0.99, inliersMask);

            std::vector<cv::DMatch> inlierMatches;
            for (size_t i = 0; i < inliersMask.size(); ++i) {
                if (inliersMask[i]) {
                    inlierMatches.push_back(goodMatches[i]);
                }
            }
            result.matches = std::move(inlierMatches);
        } else {
            result.matches = std::move(goodMatches);
        }
    
        return result;
    }
    
    std::vector<MatchData> FeatureMatcher::matchSequential(const std::vector<ImageData>& images, int neighborCount) {
        std::vector<MatchData> allMatches;

        if (images.size() < 2) {
            return allMatches;
        }
        
        for (size_t i = 0; i < images.size() - 1; ++i) {
            for (size_t j = i + 1; j <= std::min(i + neighborCount, images.size() - 1); ++j) {
                std::cout << "Matching image " << (i + 1) << " with image " << (j + 1) << "..." << std::endl;

                auto match = matchFeatures(images[i], images[j], i, j);
                if (match.isValid()) {
                    std::cout << "  Found " << match.matches.size() << " valid matches." << std::endl;
                    allMatches.push_back(std::move(match));
                } else {
                    std::cout << "  Insufficient matches (" << match.matches.size() << ")." << std::endl;
                }
            }
        }
        
        return allMatches;
    }
    
    std::vector<MatchData> FeatureMatcher::matchAllPairs(const std::vector<ImageData>& images) {
        std::vector<MatchData> allMatches;

        if (images.size() < 2) {
            return allMatches;
        }

        const size_t totalPairs = (images.size() * (images.size() - 1)) / 2;
        size_t pairCounter = 0;

        for (size_t i = 0; i < images.size() - 1; ++i) {
            for (size_t j = i + 1; j < images.size(); ++j) {
                ++pairCounter;
                std::cout << "Matching image pair " << pairCounter << "/" << totalPairs
                          << ": " << (i + 1) << " with " << (j + 1) << "..." << std::endl;

                auto match = matchFeatures(images[i], images[j], i, j);
                if (match.isValid()) {
                    std::cout << "  Found " << match.matches.size() << " valid matches." << std::endl;
                    allMatches.push_back(std::move(match));
                } else {
                    std::cout << "  Insufficient matches (" << match.matches.size() << ")." << std::endl;
                }
            }
        }

        return allMatches;
    }
}
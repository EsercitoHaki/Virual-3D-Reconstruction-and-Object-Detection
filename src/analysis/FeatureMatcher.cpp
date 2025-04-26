#include "FeatureMatcher.h"

#include <iostream>
#include <algorithm>

namespace ImageProcessing {
    FeatureMatcher::FeatureMatcher() {
        matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    }
    
    void FeatureMatcher::setMatcherFromDescriptorType(int descriptorType) {
        if (descriptorType == CV_32F) {
            matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        } else {
            matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
        }
    }
    
    MatchData FeatureMatcher::matchFeatures(const ImageData& image1, const ImageData& image2, int idx1, int idx2) {
        MatchData result;
        result.imageIdx1 = idx1;
        result.imageIdx2 = idx2;
    
        if (image1.getDescriptors().empty() || image2.getDescriptors().empty()) {
            return result;
        }
    
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(image1.getDescriptors(), image2.getDescriptors(), knnMatches, 2);
    
        std::vector<cv::DMatch> goodMatches;
        for (const auto& match : knnMatches) {
            if (match.size() >= 2 && match[0].distance < ratioTestThreshold * match[1].distance) {
                goodMatches.push_back(match[0]);
            }
        }
    
        std::sort(goodMatches.begin(), goodMatches.end(),
                [](const cv::DMatch& a, const cv::DMatch& b) {
                    return a.distance < b.distance;
                });
    
        if (goodMatches.size() > static_cast<size_t>(maxMatches)) {
            goodMatches.resize(maxMatches);
        }
    
        result.matches = goodMatches;
    
        if (goodMatches.size() >= 8) {
            std::vector<cv::Point2f> points1, points2;
            for (const auto& match : goodMatches) {
                points1.push_back(image1.getKeypoints()[match.queryIdx].pt);
                points2.push_back(image2.getKeypoints()[match.trainIdx].pt);
            }
    
            result.fundamentalMatrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3.0, 0.99);
        }
    
        return result;
    }
    
    std::vector<MatchData> FeatureMatcher::matchSequential(const std::vector<ImageData>& images, int neighborCount) {
        std::vector<MatchData> allMatches;
        
        for (size_t i = 0; i < images.size(); i++) {
            for (size_t j = i + 1; j <= std::min(i + neighborCount, images.size() - 1); j++) {
                MatchData match = matchFeatures(images[i], images[j], i, j);
                
                if (!match.matches.empty()) {
                    allMatches.push_back(match);
                    std::cout << "Matched image " << i << " with " << j 
                             << ": " << match.matches.size() << " matches" << std::endl;
                }
            }
        }
        
        return allMatches;
    }
    
    std::vector<MatchData> FeatureMatcher::matchAllPairs(const std::vector<ImageData>& images) {
        std::vector<MatchData> allMatches;
        
        for (size_t i = 0; i < images.size(); i++) {
            for (size_t j = i + 1; j < images.size(); j++) {
                MatchData match = matchFeatures(images[i], images[j], i, j);
                
                if (!match.matches.empty()) {
                    allMatches.push_back(match);
                    std::cout << "Matched image " << i << " with " << j 
                             << ": " << match.matches.size() << " matches" << std::endl;
                }
            }
        }
        
        return allMatches;
    }
}
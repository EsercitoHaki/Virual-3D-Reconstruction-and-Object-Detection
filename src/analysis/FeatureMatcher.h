#pragma once

#include "ImageData.h"
#include <opencv2/features2d.hpp>
#include <memory>

namespace ImageProcessing {
    struct MatchData {
        int imageIdx1;
        int imageIdx2;
        std::vector<cv::DMatch> matches;
        cv::Mat fundamentalMatrix;

        size_t getMatchCount() const { return matches.size(); }

        bool isValid(size_t minMatches = 20) const { return matches.size() >= minMatches; }
    };

    class FeatureMatcher {
        public:
            FeatureMatcher();

            void setMatcherFromDescriptorType(int descriptorType);

            MatchData matchFeatures(const ImageData& image1, const ImageData& image2, int idx1, int idx2);

            std::vector<MatchData> matchSequential(const std::vector<ImageData>& images, int neighborCount = 2);

            std::vector<MatchData> matchAllPairs(const std::vector<ImageData>& images);

        private:
            cv::Ptr<cv::DescriptorMatcher> matcher;
            float ratioTestThreshold = 0.75f;
            int maxMatches = 500;
    };
}
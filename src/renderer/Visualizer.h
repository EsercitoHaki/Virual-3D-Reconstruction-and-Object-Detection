#pragma once

#include "../analysis/ImageData.h"
#include "../analysis/FeatureMatcher.h"
#include "string"
#include "filesystem"

namespace fs = std::filesystem;

namespace Rendering {
    class Visualizer {
        public:
            explicit Visualizer(const std::string& outputDir);

            void visualizeKeypoints(const ImageProcessing::ImageData& imageData, size_t index);

            void visualizeMatches(const ImageProcessing::MatchData& match,
                                  const std::vector<ImageProcessing::ImageData>& images);
            
            void createMatchGraph(const std::vector<ImageProcessing::MatchData>& matches,
                                  size_t imageCount,
                                  size_t minMatches = 20);

            void visualizeAllMatches(const std::vector<ImageProcessing::ImageData>& images,
                                     const std::vector<ImageProcessing::MatchData>& matches,
                                     size_t minMatches = 20);

        private:
            std::string outputDir;
            
            void ensureOutputDirExists();
    };
}
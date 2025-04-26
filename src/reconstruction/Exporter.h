#pragma once

#include "../analysis/ImageData.h"
#include "../analysis/FeatureMatcher.h"
#include <string>
#include <vector>

namespace Reconstruction {

class Exporter {
    public:
        explicit Exporter(const std::string& outputDir);
        
        bool exportMatchesForReconstruction(
            const std::vector<ImageProcessing::ImageData>& images,
            const std::vector<ImageProcessing::MatchData>& matches,
            size_t minMatches = 20);
            
        std::string getOutputFilePath() const;

    private:
        std::string outputDir;
        std::string outputFile;
    };
}
#include "Exporter.h"
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

namespace Reconstruction {
    Exporter::Exporter(const std::string& outputDir) : outputDir(outputDir) {
        if (!fs::exists(outputDir)) {
            fs::create_directories(outputDir);
        }
        
        outputFile = outputDir + "/matches.txt";
    }

    bool Exporter::exportMatchesForReconstruction(
        const std::vector<ImageProcessing::ImageData>& images,
        const std::vector<ImageProcessing::MatchData>& matches,
        size_t minMatches) {
        
        std::ofstream output(outputFile);

        if (!output.is_open()) {
            std::cerr << "Failed to open output file for matches" << std::endl;
            return false;
        }

        output << "# Image paths:" << std::endl;
        for (size_t i = 0; i < images.size(); i++) {
            output << i << " " << images[i].getPath() << std::endl;
        }

        output << "# Matches (image1_idx image2_idx num_matches):" << std::endl;

        for (const auto& match : matches) {
            if (match.getMatchCount() >= minMatches) {
                output << match.imageIdx1 << " " << match.imageIdx2 << " " 
                    << match.matches.size() << std::endl;

                for (const auto& m : match.matches) {
                    const cv::KeyPoint& kp1 = images[match.imageIdx1].getKeypoints()[m.queryIdx];
                    const cv::KeyPoint& kp2 = images[match.imageIdx2].getKeypoints()[m.trainIdx];

                    output << kp1.pt.x << " " << kp1.pt.y << " " 
                        << kp2.pt.x << " " << kp2.pt.y << " " 
                        << m.distance << std::endl;
                }
            }
        }

        output.close();
        return true;
    }

    std::string Exporter::getOutputFilePath() const {
        return outputFile;
    }

}
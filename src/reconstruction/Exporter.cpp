#include "Exporter.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ctime>

namespace fs = std::filesystem;

constexpr float PI = 3.14159265359f;

namespace Reconstruction {
    std::string getCurrentDateTime() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
    
        std::stringstream ss;

#if defined(_WIN32) || defined(_WIN64)
        struct tm timeinfo;
        localtime_s(&timeinfo, &time);
        ss << std::put_time(&timeinfo, "%Y-%m-%d %H:%M:%S");
#else
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
#endif
        return ss.str();
    }


    Exporter::Exporter(const std::string& outputDir) : outputDir(outputDir) {
        fs::create_directories(outputDir);
        
        outputFile = outputDir + "/matches.txt";
    }

    bool Exporter::exportMatchesForReconstruction(
        const std::vector<ImageProcessing::ImageData>& images,
        const std::vector<ImageProcessing::MatchData>& matches,
        size_t minMatches) 
    {
        if (images.empty() || matches.empty()) {
            std::cerr << "[Exporter] No images or matches to export!" << std::endl;
            return false;
        }
    
        std::ofstream outFile(outputFile);
        if (!outFile) {
            std::cerr << "[Exporter] Failed to open output file: " << outputFile << std::endl;
            return false;
        }
    
        outFile << "# Feature matching reconstruction data\n";
        outFile << "# Generated on " << getCurrentDateTime() << "\n\n";
    
        // Write image list
        outFile << "# Image list\n";
        outFile << images.size() << "\n";
        for (size_t i = 0; i < images.size(); ++i) {
            const auto& img = images[i];
            const auto [width, height] = img.getDimensions();
            outFile << i << " " << img.getPath() << " "
                    << width << " " << height << " "
                    << img.getKeypointCount() << "\n";
        }
        outFile << "\n";
    
        // Write keypoints
        outFile << "# Keypoints\n";
        for (size_t i = 0; i < images.size(); ++i) {
            const auto& keypoints = images[i].getKeypoints();
            outFile << "# Image " << i << " keypoints\n";
            outFile << keypoints.size() << "\n";
            for (size_t k = 0; k < keypoints.size(); ++k) {
                const auto& kp = keypoints[k];
                outFile << k << " " << kp.pt.x << " " << kp.pt.y << " "
                        << kp.size << " " << kp.angle << " " 
                        << kp.response << " " << kp.octave << "\n";
            }
            outFile << "\n";
        }
    
        // Write matches
        int validMatchCount = 0;
        for (const auto& match : matches) {
            if (match.matches.size() >= minMatches) {
                ++validMatchCount;
            }
        }
    
        outFile << "# Matches\n";
        outFile << validMatchCount << "\n";
    
        for (const auto& match : matches) {
            if (match.matches.size() < minMatches) continue;
    
            outFile << match.imageIdx1 << " " << match.imageIdx2 << " "
                    << match.matches.size() << "\n";
            
            for (const auto& m : match.matches) {
                outFile << m.queryIdx << " " << m.trainIdx << " " << m.distance << "\n";
            }
    
            // Write Fundamental Matrix
            outFile << "# Fundamental matrix\n";
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    outFile << match.fundamentalMatrix.at<double>(i, j) << " ";
                }
                outFile << "\n";
            }
            outFile << "\n";
        }
    
        outFile.close();
        std::cout << "[Exporter] Exported matches to " << outputFile << std::endl;
    
        exportCameraPositionsPLY(images, matches, minMatches);
        return true;
    }

    void Exporter::exportCameraPositionsPLY(
        const std::vector<ImageProcessing::ImageData>& images,
        const std::vector<ImageProcessing::MatchData>& matches,
        size_t minMatches)
    {
        std::string plyFile = outputDir + "/camera_positions.ply";
        std::ofstream outFile(plyFile);
        if (!outFile) {
            std::cerr << "[Exporter] Failed to open PLY file: " << plyFile << std::endl;
            return;
        }
    
        int validEdgeCount = 0;
        for (const auto& match : matches) {
            if (match.matches.size() >= minMatches) {
                ++validEdgeCount;
            }
        }
    
        outFile << "ply\n";
        outFile << "format ascii 1.0\n";
        outFile << "element vertex " << images.size() << "\n";
        outFile << "property float x\n";
        outFile << "property float y\n";
        outFile << "property float z\n";
        outFile << "property uchar red\n";
        outFile << "property uchar green\n";
        outFile << "property uchar blue\n";
        outFile << "element edge " << validEdgeCount << "\n";
        outFile << "property int vertex1\n";
        outFile << "property int vertex2\n";
        outFile << "end_header\n";
    
        for (size_t i = 0; i < images.size(); ++i) {
            float angle = static_cast<float>(i) / images.size() * 2.0f * PI;
            float x = std::cos(angle) * 10.0f;
            float y = 0.0f;
            float z = std::sin(angle) * 10.0f;
    
            int r = 255 * (i % 3 == 0);
            int g = 255 * (i % 3 == 1);
            int b = 255 * (i % 3 == 2);
    
            outFile << x << " " << y << " " << z << " "
                    << r << " " << g << " " << b << "\n";
        }
    
        for (const auto& match : matches) {
            if (match.matches.size() >= minMatches) {
                outFile << match.imageIdx1 << " " << match.imageIdx2 << "\n";
            }
        }
    
        outFile.close();
        std::cout << "[Exporter] Exported camera positions to " << plyFile << std::endl;
    }
    

    std::string Exporter::getOutputFilePath() const {
        return outputFile;
    }

}
#include "Visualizer.h"

#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>

namespace Rendering {
    Visualizer::Visualizer(const std::string& outputDir) : outputDir(outputDir) {
        ensureOutputDirExists();
    }

    void Visualizer::ensureOutputDirExists() {
        fs::create_directories(outputDir);
        fs::create_directories(outputDir + "/keypoints");
        fs::create_directories(outputDir + "/matches");
        fs::create_directories(outputDir + "/graph");
    }

    void Visualizer::visualizeKeypoints(const ImageProcessing::ImageData& imageData, size_t index) {
        cv::Mat keypointsImage;
        cv::drawKeypoints(imageData.getImage(), imageData.getKeypoints(), keypointsImage,
                          cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        std::string filename = outputDir + "/keypoints/keypoints_" + std::to_string(index + 1) + ".jpg";
        if (cv::imwrite(filename, keypointsImage)) {
            std::cout << "[Visualizer] Saved keypoints visualization: " << filename << std::endl;
        } else {
            std::cerr << "[Visualizer] Failed to save keypoints image: " << filename << std::endl;
        }
    }

    void Visualizer::visualizeMatches(const ImageProcessing::MatchData& match,
                                      const std::vector<ImageProcessing::ImageData>& images) {
        if (match.imageIdx1 >= images.size() || match.imageIdx2 >= images.size()) {
            std::cerr << "[Visualizer] Invalid image indices in match data." << std::endl;
            return;
        }

        cv::Mat imgMatches;
        cv::drawMatches(
            images[match.imageIdx1].getImage(), images[match.imageIdx1].getKeypoints(),
            images[match.imageIdx2].getImage(), images[match.imageIdx2].getKeypoints(),
            match.matches, imgMatches,
            cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
            std::vector<char>(), cv::DrawMatchesFlags::DEFAULT
        );

        std::string matchText = "Matches: " + std::to_string(match.matches.size());
        cv::putText(imgMatches, matchText, cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);

        std::string filename = outputDir + "/matches/matches_" + 
                               std::to_string(match.imageIdx1 + 1) + "_" + 
                               std::to_string(match.imageIdx2 + 1) + ".jpg";

        if (cv::imwrite(filename, imgMatches)) {
            std::cout << "[Visualizer] Saved matches visualization: " << filename << std::endl;
        } else {
            std::cerr << "[Visualizer] Failed to save matches image: " << filename << std::endl;
        }
    }

    void Visualizer::createMatchGraph(const std::vector<ImageProcessing::MatchData>& matches,
                                      size_t imageCount,
                                      size_t minMatches) {
        const int width = 1000, height = 800;
        const int nodeRadius = 30;
        const int margin = 50;

        cv::Mat graph(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

        std::vector<cv::Point> nodePositions;
        double circleRadius = std::min(width, height) / 2 - margin - nodeRadius;
        cv::Point center(width / 2, height / 2);

        for (size_t i = 0; i < imageCount; ++i) {
            double angle = 2 * CV_PI * i / imageCount;
            int x = static_cast<int>(center.x + circleRadius * cos(angle));
            int y = static_cast<int>(center.y + circleRadius * sin(angle));
            nodePositions.emplace_back(x, y);
        }

        for (const auto& match : matches) {
            if (match.matches.size() >= minMatches) {
                if (match.imageIdx1 >= imageCount || match.imageIdx2 >= imageCount) continue;
                cv::Point p1 = nodePositions[match.imageIdx1];
                cv::Point p2 = nodePositions[match.imageIdx2];

                int thickness = std::min(5, std::max(1, int(match.matches.size() / 20)));
                cv::line(graph, p1, p2, cv::Scalar(150, 150, 150), thickness);

                cv::Point mid((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
                cv::putText(graph, std::to_string(match.matches.size()), mid,
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            }
        }

        for (size_t i = 0; i < nodePositions.size(); ++i) {
            cv::circle(graph, nodePositions[i], nodeRadius, cv::Scalar(100, 100, 255), -1);
            cv::circle(graph, nodePositions[i], nodeRadius, cv::Scalar(0, 0, 0), 2);

            std::string label = std::to_string(i + 1);
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
            cv::Point textPos(nodePositions[i].x - textSize.width / 2, 
                              nodePositions[i].y + textSize.height / 2);

            cv::putText(graph, label, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.8, 
                        cv::Scalar(255, 255, 255), 2);
        }

        std::string graphFilename = outputDir + "/graph/match_graph.png";
        if (cv::imwrite(graphFilename, graph)) {
            std::cout << "[Visualizer] Saved match graph image: " << graphFilename << std::endl;
        } else {
            std::cerr << "[Visualizer] Failed to save graph image." << std::endl;
        }

        std::ofstream dotFile(outputDir + "/graph/match_graph.dot");
        if (dotFile.is_open()) {
            dotFile << "graph MatchGraph {\n";
            for (size_t i = 0; i < imageCount; ++i) {
                dotFile << "    " << (i + 1) << ";\n";
            }
            for (const auto& match : matches) {
                if (match.matches.size() >= minMatches) {
                    dotFile << "    " << (match.imageIdx1 + 1) << " -- " 
                            << (match.imageIdx2 + 1) 
                            << " [label=" << match.matches.size() << "];\n";
                }
            }
            dotFile << "}\n";
            dotFile.close();
            std::cout << "[Visualizer] Saved Graphviz DOT file." << std::endl;
        } else {
            std::cerr << "[Visualizer] Failed to save DOT file." << std::endl;
        }
    }

    void Visualizer::visualizeAllMatches(const std::vector<ImageProcessing::ImageData>& images,
                                         const std::vector<ImageProcessing::MatchData>& matches,
                                         size_t minMatches) {
        for (size_t i = 0; i < images.size(); i++) {
            visualizeKeypoints(images[i], i);
        }

        for (const auto& match : matches) {
            if (match.getMatchCount() >= minMatches) {
                visualizeMatches(match, images);
            }
        }

        createMatchGraph(matches, images.size(), minMatches);
    }
}
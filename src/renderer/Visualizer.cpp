#include "Visualizer.h"
#include <opencv2/highgui.hpp>
#include <iostream>

namespace Rendering {
    Visualizer::Visualizer(const std::string& outputDir) : outputDir(outputDir) {
        ensureOutputDirExists();
    }

    void Visualizer::ensureOutputDirExists() {
        if (!fs::exists(outputDir)) {
            fs::create_directories(outputDir);
        }
    }

    void Visualizer::visualizeKeypoints(const ImageProcessing::ImageData& imageData, size_t index) {
        cv::Mat keypointsImage;
        cv::drawKeypoints(imageData.getImage(), imageData.getKeypoints(), keypointsImage,
                          cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        std::string filename = outputDir + "/keypoints_" + std::to_string(index) + ".jpg";
        cv::imwrite(filename, keypointsImage);
    }

    void Visualizer::visualizeMatches(const ImageProcessing::MatchData& match,
                                      const std::vector<ImageProcessing::ImageData>& images) {
        if (!match.isValid()) {
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

        std::string matchText = "Matches: " + std::to_string(match.matches.size()) + 
                                " between images " + std::to_string(match.imageIdx1) + 
                                " and " + std::to_string(match.imageIdx2);

        cv::putText(imgMatches, matchText, cv::Point(20, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

        std::string filename = outputDir + "/matches_" + 
                               std::to_string(match.imageIdx1) + "_" + 
                               std::to_string(match.imageIdx2) + ".jpg";
                
        cv::imwrite(filename, imgMatches);
    }

    void Visualizer::createMatchGraph(const std::vector<ImageProcessing::MatchData>& matches,
                                      size_t imageCount,
                                      size_t minMatches) {
        const int graphSize = 800;

        cv::Mat matchGraph(graphSize, graphSize, CV_8UC3, cv::Scalar(255, 255, 255));

        std::vector<cv::Point> nodePositions;

        for (size_t i = 0; i < imageCount; i++) {
            float angle = 2 * CV_PI * i / imageCount;
            int radius = graphSize / 3;
            cv::Point pos(
                graphSize / 2 + radius * cos(angle),
                graphSize / 2 + radius * sin(angle)
            );

            nodePositions.push_back(pos);

            cv::circle(matchGraph, pos, 20, cv::Scalar(0, 0, 0), -1);
            cv::putText(matchGraph, std::to_string(i),
                        cv::Point(pos.x - 5, pos.y + 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        for (const auto& match : matches) {
            if (match.getMatchCount() >= minMatches) {
                cv::Point p1 = nodePositions[match.imageIdx1];
                cv::Point p2 = nodePositions[match.imageIdx2];

                int thickness = std::min(8, static_cast<int>(1 + match.getMatchCount() / 50));
                cv::line(matchGraph, p1, p2, cv::Scalar(0, 0, 255), thickness);

                cv::Point labelPos((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
                cv::putText(matchGraph, std::to_string(match.getMatchCount()),
                            labelPos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            }
        }

        cv::imwrite(outputDir + "/match_graph.jpg", matchGraph);
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
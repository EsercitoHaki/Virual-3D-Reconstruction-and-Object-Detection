#pragma once

#include "../analysis/ImageData.h"
#include "../analysis/FeatureMatcher.h"
#include "CameraModel.h"

#include "opencv2/core.hpp"
#include "vector"
#include <unordered_map>

namespace SFM {
    struct CameraPose {
        cv::Mat rotation;
        cv::Mat translation;
        cv::Mat rotationVec;
        cv::Mat projMatrix;
        bool isEstimated = false;

        void updateProjectionMatrix(const cv::Mat& cameraMatrix);
        
        bool operator==(const CameraPose& other) const {
            return cv::norm(rotation - other.rotation) < 1e-6 && 
                   cv::norm(translation - other.translation) < 1e-6 &&
                   isEstimated == other.isEstimated;
        }
    };

    class PoseEstimator {
        public:
            PoseEstimator(const CameraModel& cameraModel);

            bool initializePoses(
                const std::vector<ImageProcessing::ImageData>& images,
                const std::vector<ImageProcessing::MatchData>& matches
            );

            bool estimatePoseFromPoints(
                size_t imageIdx,
                const std::vector<cv::Point2f>& points2D,
                const std::vector<cv::Point3f>& points3D
            );

            CameraPose getCameraPose(size_t imageIdx) const;

            const std::unordered_map<size_t, CameraPose>& getCameraPoses() const;

            std::vector<CameraPose> getAllCameraPoses() const;

            std::vector<size_t> getEstimatedViewIndices() const;

            bool savePoses(const std::string& filename) const;

            std::pair<size_t, size_t> findInitialImagePair(
                const std::vector<ImageProcessing::MatchData>& matches);


        private:
            bool estimateRelativePose(
                const std::vector<cv::DMatch>& matches,
                const std::vector<cv::KeyPoint>& keypoints1,
                const std::vector<cv::KeyPoint>& keypoints2,
                std::vector<uchar>& mask,
                cv::Mat& R,
                cv::Mat& t
            );
            
            CameraModel cameraModel;
            std::unordered_map<size_t, CameraPose> cameraPoses;

    };
}
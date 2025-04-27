#include "PoseEstimator.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace SFM {

    PoseEstimator::PoseEstimator(const CameraModel& cameraModel) 
        : cameraModel(cameraModel) {
    }

    bool PoseEstimator::initializePoses(
        const std::vector<ImageProcessing::ImageData>& images,
    const std::vector<ImageProcessing::MatchData>& matches) {
    
        auto initialPair = findInitialImagePair(matches);
        size_t idx1 = initialPair.first;
        size_t idx2 = initialPair.second;
        
        const ImageProcessing::MatchData* pairMatch = nullptr;
        for (const auto& match : matches) {
            if ((match.imageIdx1 == idx1 && match.imageIdx2 == idx2) ||
                (match.imageIdx1 == idx2 && match.imageIdx2 == idx1)) {
                pairMatch = &match;
                break;
            }
        }
        
        if (!pairMatch) {
            return false;
        }
        
        std::vector<cv::DMatch> goodMatches = pairMatch->matches;
        const std::vector<cv::KeyPoint>& keypoints1 = 
            (pairMatch->imageIdx1 == idx1) ? 
            images[idx1].getKeypoints() : images[idx2].getKeypoints();
        const std::vector<cv::KeyPoint>& keypoints2 = 
            (pairMatch->imageIdx1 == idx1) ? 
            images[idx2].getKeypoints() : images[idx1].getKeypoints();
        
        std::vector<uchar> mask;
        cv::Mat R, t;
        if (!estimateRelativePose(goodMatches, keypoints1, keypoints2, mask, R, t)) {
            return false;
        }
        
        CameraPose pose1;
        pose1.rotation = cv::Mat::eye(3, 3, CV_64F);
        pose1.translation = cv::Mat::zeros(3, 1, CV_64F);
        
        cv::Mat rotVec;
        cv::Rodrigues(pose1.rotation, rotVec);
        rotVec.convertTo(pose1.rotationVec, CV_64F);
        
        pose1.isEstimated = true;
        
        // In thêm thông tin về rotation vector
        std::cout << "Pose1 RotVec type: " << pose1.rotationVec.type() 
                  << ", size: " << pose1.rotationVec.size() << std::endl;
        
        pose1.updateProjectionMatrix(cameraModel.getCameraMatrix());
        cameraPoses[idx1] = pose1;
        
        CameraPose pose2;
        // Chuyển đổi R và t sang CV_64F và đảm bảo đúng kích thước
        R.convertTo(pose2.rotation, CV_64F);
        t.convertTo(pose2.translation, CV_64F);
        
        // Đảm bảo translation là vector cột
        if (pose2.translation.rows == 1 && pose2.translation.cols == 3) {
            pose2.translation = pose2.translation.t();
        }
        
        cv::Mat rotVec2;
        cv::Rodrigues(pose2.rotation, rotVec2);
        rotVec2.convertTo(pose2.rotationVec, CV_64F);
        
        pose2.isEstimated = true;
        
        // In thêm thông tin về rotation vector
        std::cout << "Pose2 RotVec type: " << pose2.rotationVec.type() 
                  << ", size: " << pose2.rotationVec.size() << std::endl;
        
        pose2.updateProjectionMatrix(cameraModel.getCameraMatrix());
        cameraPoses[idx2] = pose2;
        
        return true;
    }

    bool PoseEstimator::estimateRelativePose(
        const std::vector<cv::DMatch>& matches,
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        std::vector<uchar>& mask,
        cv::Mat& R,
        cv::Mat& t) {
        
        std::vector<cv::Point2f> points1, points2;
        for (const auto& match : matches) {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }
        
        cv::Mat K = cameraModel.getCameraMatrix();
        
        cv::Mat E = cv::findEssentialMat(
            points1, points2, K, 
            cv::RANSAC, 0.999, 1.0, mask);
        
        if (E.empty()) {
            return false;
        }
        
        int inliers = cv::recoverPose(E, points1, points2, K, R, t, mask);
        
        return inliers >= std::max(20, static_cast<int>(matches.size() * 0.3));
    }

    bool PoseEstimator::estimatePoseFromPoints(
        size_t imageIdx,
        const std::vector<cv::Point2f>& points2D,
        const std::vector<cv::Point3f>& points3D) {
        
            if (points2D.size() < 4 || points2D.size() != points3D.size()) {
                return false;
            }
            
            cv::Mat K = cameraModel.getCameraMatrix();
            cv::Mat distCoeffs = cameraModel.getDistortionCoeffs();
            
            // Đảm bảo ma trận được khởi tạo đúng kiểu
            K.convertTo(K, CV_64F);
            distCoeffs.convertTo(distCoeffs, CV_64F);
            
            cv::Mat rvec, tvec;
            std::vector<int> inliers;
            
            bool success = cv::solvePnPRansac(
                points3D, points2D, K, distCoeffs, 
                rvec, tvec, false, 100, 8.0, 0.99, inliers);
            
            if (!success || inliers.size() < 10) {
                return false;
            }
            
            CameraPose pose;
            // Đảm bảo chuyển đổi đúng kiểu
            rvec.convertTo(pose.rotationVec, CV_64F);
            tvec.convertTo(pose.translation, CV_64F);
            
            cv::Rodrigues(pose.rotationVec, pose.rotation);
            pose.isEstimated = true;
            pose.updateProjectionMatrix(K);
            
            cameraPoses[imageIdx] = pose;
            
            return true;
    }

    CameraPose PoseEstimator::getCameraPose(size_t imageIdx) const {
        auto it = cameraPoses.find(imageIdx);
        if (it != cameraPoses.end()) {
            return it->second;
        }
        return CameraPose();
    }

    std::vector<CameraPose> PoseEstimator::getAllCameraPoses() const {
        std::vector<CameraPose> poses;
        for (const auto& pair : cameraPoses) {
            poses.push_back(pair.second);
        }
        return poses;
    }

    std::vector<size_t> PoseEstimator::getEstimatedViewIndices() const {
        std::vector<size_t> indices;
        for (const auto& pair : cameraPoses) {
            if (pair.second.isEstimated) {
                indices.push_back(pair.first);
            }
        }
        return indices;
    }

    std::pair<size_t, size_t> PoseEstimator::findInitialImagePair(
        const std::vector<ImageProcessing::MatchData>& matches) {
        
        size_t bestIdx1 = 0;
        size_t bestIdx2 = 1;
        int bestScore = -1;
        
        for (const auto& match : matches) {
            int numMatches = match.matches.size();
            if (numMatches > bestScore) {
                bestScore = numMatches;
                bestIdx1 = match.imageIdx1;
                bestIdx2 = match.imageIdx2;
            }
        }
        
        return {bestIdx1, bestIdx2};
    }

    bool PoseEstimator::savePoses(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        file << "# Camera poses in format: image_idx R_vec[3] t[3]" << std::endl;
        file << "# Where R_vec is Rodrigues rotation vector" << std::endl;
        
        for (const auto& pair : cameraPoses) {
            if (!pair.second.isEstimated) continue;
            
            file << pair.first << " ";
            
            for (int i = 0; i < 3; i++) {
                file << pair.second.rotationVec.at<double>(i, 0) << " ";
            }
            
            for (int i = 0; i < 3; i++) {
                file << pair.second.translation.at<double>(i, 0) << " ";
            }
            
            file << std::endl;
        }
        
        file.close();
        return true;
    }

    void CameraPose::updateProjectionMatrix(const cv::Mat& cameraMatrix) {
        // Chuyển đổi ma trận sang CV_64F một cách chắc chắn
        cv::Mat R64, t64, K64;
        cv::Mat(rotation).convertTo(R64, CV_64F);
        cv::Mat(translation).convertTo(t64, CV_64F);
        cv::Mat(cameraMatrix).convertTo(K64, CV_64F);

        // Đảm bảo translation là vector cột 3x1
        if (t64.rows != 3 || t64.cols != 1) {
            t64 = t64.reshape(0, 3);
        }

        // Khởi tạo projection matrix
        projMatrix = cv::Mat::zeros(3, 4, CV_64F);
        
        // Sao chép rotation
        R64.copyTo(projMatrix(cv::Rect(0, 0, 3, 3)));
        
        // Sao chép translation
        t64.copyTo(projMatrix(cv::Rect(3, 0, 1, 3)));

        // Nhân ma trận một cách an toàn
        cv::Mat result;
        cv::Mat temp = K64 * projMatrix;
        temp.convertTo(result, CV_64F);
        projMatrix = result;

        // Debug
        std::cout << "Final Projection Matrix:" << std::endl;
        std::cout << "Type: " << projMatrix.type() 
                << ", Size: " << projMatrix.rows << "x" << projMatrix.cols 
                << ", Channels: " << projMatrix.channels() << std::endl;
    }

    const std::unordered_map<size_t, CameraPose>& PoseEstimator::getCameraPoses() const {
        return cameraPoses;
    }

}
#include "Triangulator.h"

#include <opencv2/calib3d.hpp>
#include <set>
#include <iostream>

namespace SFM {
    Triangulator::Triangulator(const CameraModel& cameraModel) : cameraModel(cameraModel) {
    }

    size_t Triangulator::triangulateInitialPair(
        const std::vector<ImageProcessing::ImageData>& images,
        const std::pair<size_t, size_t>& initialPair,
        const std::vector<ImageProcessing::MatchData>& matches,
        const std::unordered_map<size_t, CameraPose>& cameraPoses) {
        
        const ImageProcessing::MatchData* pairMatch = nullptr;
        for (const auto& match : matches) {
            if ((match.imageIdx1 == initialPair.first && match.imageIdx2 == initialPair.second) ||
                (match.imageIdx1 == initialPair.second && match.imageIdx2 == initialPair.first)) {
                pairMatch = &match;
                break;
            }
        }
        
        if (!pairMatch) {
            return 0;
        }
        
        auto it1 = cameraPoses.find(initialPair.first);
        auto it2 = cameraPoses.find(initialPair.second);
        if (it1 == cameraPoses.end() || it2 == cameraPoses.end()) {
            return 0;
        }
        
        const CameraPose& pose1 = it1->second;
        const CameraPose& pose2 = it2->second;
        
        size_t idx1 = pairMatch->imageIdx1;
        size_t idx2 = pairMatch->imageIdx2;
        bool swapped = (idx1 != initialPair.first);
        
        std::vector<cv::Point2f> points1, points2;
        std::vector<size_t> keypointIndices1, keypointIndices2;
        
        for (const auto& match : pairMatch->matches) {
            size_t queryIdx = swapped ? match.trainIdx : match.queryIdx;
            size_t trainIdx = swapped ? match.queryIdx : match.trainIdx;
            
            points1.push_back(images[initialPair.first].getKeypoints()[queryIdx].pt);
            points2.push_back(images[initialPair.second].getKeypoints()[trainIdx].pt);
            
            keypointIndices1.push_back(queryIdx);
            keypointIndices2.push_back(trainIdx);
        }
        
        cv::Mat K = cameraModel.getCameraMatrix();
        cv::Mat distCoeffs = cameraModel.getDistortionCoeffs();
        
        // Chuyển đổi sang CV_64F
        K.convertTo(K, CV_64F);
        distCoeffs.convertTo(distCoeffs, CV_64F);
        
        if (!distCoeffs.empty() && cv::countNonZero(distCoeffs) > 0) {
            cv::undistortPoints(points1, points1, K, distCoeffs, cv::noArray(), K);
            cv::undistortPoints(points2, points2, K, distCoeffs, cv::noArray(), K);
        }

        // Debug thông tin ma trận
        auto debugMatrixInfo = [](const cv::Mat& mat, const std::string& name) {
            std::cout << name << " Matrix Info:" << std::endl;
            std::cout << "Type: " << mat.type() << std::endl;
            std::cout << "Size: " << mat.rows << "x" << mat.cols << std::endl;
            std::cout << "Channels: " << mat.channels() << std::endl;
            std::cout << "Depth: " << mat.depth() << std::endl;
            
            // In một số giá trị đầu tiên để kiểm tra
            std::cout << "First few values:" << std::endl;
            for (int i = 0; i < std::min(3, mat.rows); ++i) {
                for (int j = 0; j < std::min(4, mat.cols); ++j) {
                    std::cout << mat.at<double>(i, j) << " ";
                }
                std::cout << std::endl;
            }
        };

        // Chuyển các điểm sang Point2d để sử dụng với CV_64F
        std::vector<cv::Point2d> points1_64d, points2_64d;
        for (const auto& pt : points1) {
            points1_64d.push_back(cv::Point2d(pt.x, pt.y));
        }
        for (const auto& pt : points2) {
            points2_64d.push_back(cv::Point2d(pt.x, pt.y));
        }
    
        // Đảm bảo projection matrices đều là CV_64F
        cv::Mat projMatrix1_64F = cv::Mat::zeros(3, 4, CV_64F);
        cv::Mat projMatrix2_64F = cv::Mat::zeros(3, 4, CV_64F);
    
        // Sao chép giá trị từ projection matrix gốc, đảm bảo kiểu CV_64F
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                projMatrix1_64F.at<double>(i, j) = pose1.projMatrix.at<double>(i, j);
                projMatrix2_64F.at<double>(i, j) = pose2.projMatrix.at<double>(i, j);
            }
        }
    
        // Debug: In thông tin chi tiết ma trận
        std::cout << "Projection Matrix 1 Type: " << projMatrix1_64F.type() 
                  << ", Size: " << projMatrix1_64F.rows << "x" << projMatrix1_64F.cols << std::endl;
        std::cout << "Projection Matrix 2 Type: " << projMatrix2_64F.type() 
                  << ", Size: " << projMatrix2_64F.rows << "x" << projMatrix2_64F.cols << std::endl;
    
        // Kiểm tra và in thông tin điểm
        std::cout << "Points1_64d Size: " << points1_64d.size() << std::endl;
        std::cout << "Points2_64d Size: " << points2_64d.size() << std::endl;
    
        cv::Mat points4D;
        try {
            // Thực hiện triangulation với tất cả dữ liệu là CV_64F
            cv::triangulatePoints(projMatrix1_64F, projMatrix2_64F, points1_64d, points2_64d, points4D);
            
            // Chuyển points4D sang CV_64F nếu cần
            cv::Mat points4D_64F;
            points4D.convertTo(points4D_64F, CV_64F);
            
            size_t addedPoints = 0;
            points3D.clear();
            
            for (int i = 0; i < points4D_64F.cols; i++) {
                double w = points4D_64F.at<double>(3, i);
                if (std::abs(w) < 1e-10) continue;
                
                cv::Point3d point3d(
                    points4D_64F.at<double>(0, i) / w,
                    points4D_64F.at<double>(1, i) / w,
                    points4D_64F.at<double>(2, i) / w
                );
                
                // Kiểm tra điểm phía trước camera
                if (point3d.z <= 0) continue;
                
                cv::Mat pointMat = (cv::Mat_<double>(4, 1) << point3d.x, point3d.y, point3d.z, 1.0);
                cv::Mat transformedPoint = pose2.rotation * cv::Mat(point3d) + pose2.translation;
                if (transformedPoint.at<double>(2, 0) <= 0) continue;
                
                Point3D newPoint;
                newPoint.position = cv::Point3f(point3d.x, point3d.y, point3d.z); // Chuyển từ Point3d sang Point3f
                
                const cv::Point2f& pt = points1[i];
                if (pt.x >= 0 && pt.x < images[initialPair.first].getImage().cols &&
                    pt.y >= 0 && pt.y < images[initialPair.first].getImage().rows) {
                    cv::Vec3b color = images[initialPair.first].getImage().at<cv::Vec3b>(cv::Point(pt.x, pt.y));
                    newPoint.color = color;
                } else {
                    newPoint.color = cv::Vec3b(255, 255, 255); 
                }
                
                newPoint.observations[initialPair.first] = keypointIndices1[i];
                newPoint.observations[initialPair.second] = keypointIndices2[i];
                
                points3D.push_back(newPoint);
                addedPoints++;
            }
            
            return addedPoints;
        } catch (const cv::Exception& e) {
            std::cerr << "Triangulation Error: " << e.what() << std::endl;
            return 0;
        }
    }

    size_t Triangulator::triangulateNewView(
        const std::vector<ImageProcessing::ImageData>& images,
        size_t imageIdx,
        const std::vector<ImageProcessing::MatchData>& matches,
        const std::unordered_map<size_t, CameraPose>& cameraPoses,
        const std::vector<size_t>& registeredViews) {
        
        auto it = cameraPoses.find(imageIdx);
        if (it == cameraPoses.end() || !it->second.isEstimated) {
            return 0;
        }
        
        const CameraPose& newCameraPose = it->second;
        size_t addedPoints = 0;
        
        for (size_t regViewIdx : registeredViews) {
            if (regViewIdx == imageIdx) continue;
            
            const ImageProcessing::MatchData* viewMatch = nullptr;
            for (const auto& match : matches) {
                if ((match.imageIdx1 == imageIdx && match.imageIdx2 == regViewIdx) ||
                    (match.imageIdx1 == regViewIdx && match.imageIdx2 == imageIdx)) {
                    viewMatch = &match;
                    break;
                }
            }
            
            if (!viewMatch) continue;
            
            auto regPoseIt = cameraPoses.find(regViewIdx);
            if (regPoseIt == cameraPoses.end() || !regPoseIt->second.isEstimated) {
                continue;
            }
            
            const CameraPose& regCameraPose = regPoseIt->second;
            
            bool newViewIsFirst = (viewMatch->imageIdx1 == imageIdx);
            
            std::vector<cv::Point2f> pointsNew, pointsReg;
            std::vector<size_t> keypointIndicesNew, keypointIndicesReg;
            
            for (const auto& match : viewMatch->matches) {
                size_t newViewKeypointIdx = newViewIsFirst ? match.queryIdx : match.trainIdx;
                size_t regViewKeypointIdx = newViewIsFirst ? match.trainIdx : match.queryIdx;
                
                pointsNew.push_back(images[imageIdx].getKeypoints()[newViewKeypointIdx].pt);
                pointsReg.push_back(images[regViewIdx].getKeypoints()[regViewKeypointIdx].pt);
                
                keypointIndicesNew.push_back(newViewKeypointIdx);
                keypointIndicesReg.push_back(regViewKeypointIdx);
            }
            
            cv::Mat K = cameraModel.getCameraMatrix();
            cv::Mat distCoeffs = cameraModel.getDistortionCoeffs();
            
            // Chuyển đổi kiểu dữ liệu sang CV_64F
            K.convertTo(K, CV_64F);
            distCoeffs.convertTo(distCoeffs, CV_64F);
            
            if (!distCoeffs.empty() && cv::countNonZero(distCoeffs) > 0) {
                cv::undistortPoints(pointsNew, pointsNew, K, distCoeffs, cv::noArray(), K);
                cv::undistortPoints(pointsReg, pointsReg, K, distCoeffs, cv::noArray(), K);
            }
            
            // Chuyển các điểm sang CV_64F
            std::vector<cv::Point2d> pointsNew_64d, pointsReg_64d;
            for (const auto& pt : pointsNew) {
                pointsNew_64d.push_back(cv::Point2d(pt.x, pt.y));
            }
            for (const auto& pt : pointsReg) {
                pointsReg_64d.push_back(cv::Point2d(pt.x, pt.y));
            }

            // Chuyển đổi projection matrices sang CV_64F
            cv::Mat projMatrixNew_64F = cv::Mat::zeros(3, 4, CV_64F);
            cv::Mat projMatrixReg_64F = cv::Mat::zeros(3, 4, CV_64F);

            // Sao chép giá trị từ projection matrix gốc
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 4; ++j) {
                    projMatrixNew_64F.at<double>(i, j) = newCameraPose.projMatrix.at<double>(i, j);
                    projMatrixReg_64F.at<double>(i, j) = regCameraPose.projMatrix.at<double>(i, j);
                }
            }

            cv::Mat points4D;
            try {
                // Triangulate với tất cả dữ liệu là CV_64F
                cv::triangulatePoints(projMatrixNew_64F, projMatrixReg_64F, 
                                      pointsNew_64d, pointsReg_64d, points4D);
                
                // Chuyển points4D sang CV_64F nếu cần
                cv::Mat points4D_64F;
                points4D.convertTo(points4D_64F, CV_64F);
                
                for (int i = 0; i < points4D_64F.cols; i++) {
                    double w = points4D_64F.at<double>(3, i);
                    if (std::abs(w) < 1e-10) continue;
                    
                    cv::Point3d point3d(
                        points4D_64F.at<double>(0, i) / w,
                        points4D_64F.at<double>(1, i) / w,
                        points4D_64F.at<double>(2, i) / w
                    );
                    
                    if (point3d.z <= 0) continue;
                    
                    cv::Mat transformedPoint = regCameraPose.rotation * cv::Mat(point3d) + regCameraPose.translation;
                    if (transformedPoint.at<double>(2, 0) <= 0) continue;
                    
                    bool pointExists = false;
                    for (Point3D& existingPoint : points3D) {
                        auto it = existingPoint.observations.find(regViewIdx);
                        if (it != existingPoint.observations.end() && it->second == keypointIndicesReg[i]) {
                            existingPoint.observations[imageIdx] = keypointIndicesNew[i];
                            pointExists = true;
                            break;
                        }
                    }
                    
                    if (!pointExists) {
                        Point3D newPoint;
                        newPoint.position = cv::Point3f(point3d.x, point3d.y, point3d.z); // Chuyển từ Point3d sang Point3f
                        
                        const cv::Point2f& pt = pointsNew[i];
                        if (pt.x >= 0 && pt.x < images[imageIdx].getImage().cols &&
                            pt.y >= 0 && pt.y < images[imageIdx].getImage().rows) {
                            cv::Vec3b color = images[imageIdx].getImage().at<cv::Vec3b>(cv::Point(pt.x, pt.y));
                            newPoint.color = color;
                        } else {
                            newPoint.color = cv::Vec3b(255, 255, 255);
                        }
                        
                        newPoint.observations[imageIdx] = keypointIndicesNew[i];
                        newPoint.observations[regViewIdx] = keypointIndicesReg[i];
                        
                        points3D.push_back(newPoint);
                        addedPoints++;
                    }
                }
            } catch (const cv::Exception& e) {
                std::cerr << "Triangulation Error: " << e.what() << std::endl;
                continue; // Tiếp tục với view tiếp theo
            }
        }
        
        return addedPoints;
    }

    const std::vector<Point3D>& Triangulator::getPoints() const {
        return points3D;
    }
}
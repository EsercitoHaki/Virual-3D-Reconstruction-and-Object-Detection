#include "Sfm.h"

namespace SFM {
    SfMReconstructor::SfMReconstructor(
        const std::vector<ImageProcessing::ImageData>& images,
        const std::vector<ImageProcessing::MatchData>& matches,
        const CameraModel& cameraModel)
        : images(images), matches(matches), 
        poseEstimator(cameraModel), 
        triangulator(cameraModel) {}

    bool SfMReconstructor::reconstruct() {
        // Bước 1: Khởi tạo pose của hai ảnh đầu tiên
        if (!poseEstimator.initializePoses(images, matches)) {
            std::cerr << "Failed to initialize camera poses" << std::endl;
            return false;
        }
        
        // Bước 2: Triangulate điểm 3D từ hai ảnh đầu tiên
        auto initialPair = poseEstimator.findInitialImagePair(matches);
        
        // Chuyển đổi vector sang unordered_map
        std::unordered_map<size_t, CameraPose> cameraPoses;
        auto allCameraPoses = poseEstimator.getAllCameraPoses();
        std::cout << "Debug: Camera Poses Information" << std::endl;
        const auto& existingPoses = poseEstimator.getCameraPoses();

        for (const auto& [idx, pose] : cameraPoses) {
            std::cout << "Image Index: " << idx << std::endl;
            std::cout << "Rotation Matrix Type: " << pose.rotation.type() 
                      << ", Size: " << pose.rotation.size() << std::endl;
            std::cout << "Translation Matrix Type: " << pose.translation.type() 
                      << ", Size: " << pose.translation.size() << std::endl;
            std::cout << "Projection Matrix Type: " << pose.projMatrix.type() 
                      << ", Size: " << pose.projMatrix.size() << std::endl;
        }

        for (size_t i = 0; i < allCameraPoses.size(); ++i) {
            for (const auto& [index, pose] : existingPoses) {
                // So sánh giá trị của camera pose
                if (pose.rotation.at<double>(0,0) == allCameraPoses[i].rotation.at<double>(0,0) &&
                    pose.translation.at<double>(0,0) == allCameraPoses[i].translation.at<double>(0,0)) {
                    cameraPoses[index] = allCameraPoses[i];
                    break;
                }
            }
        }

        size_t initialPointsCount = triangulator.triangulateInitialPair(
            images, initialPair, matches, cameraPoses
        );

        if (initialPointsCount == 0) {
            std::cerr << "Failed to triangulate initial point cloud" << std::endl;
            return false;
        }

        std::cout << "Triangulated " << initialPointsCount << " initial points" << std::endl;

        // Bước 3: Mở rộng reconstruction bằng cách thêm dần các view mới
        std::vector<size_t> registeredViews = {initialPair.first, initialPair.second};
        
        for (size_t imageIdx = 0; imageIdx < images.size(); ++imageIdx) {
            // Bỏ qua các view đã được xử lý
            if (std::find(registeredViews.begin(), registeredViews.end(), imageIdx) 
                != registeredViews.end()) {
                continue;
            }

            // Thử estimate pose cho view mới
            std::vector<cv::Point2f> points2D;
            std::vector<cv::Point3f> points3D;

            // Tìm các điểm 2D và 3D tương ứng từ các view đã đăng ký
            for (size_t regViewIdx : registeredViews) {
                const ImageProcessing::MatchData* match = findMatchBetweenImages(imageIdx, regViewIdx);
                if (!match) continue;

                // Lấy các điểm 3D tương ứng với điểm đặc trưng trong view đã đăng ký
                for (const auto& m : match->matches) {
                    size_t keypointIdx1 = (match->imageIdx1 == imageIdx) ? m.queryIdx : m.trainIdx;
                    size_t keypointIdx2 = (match->imageIdx1 == imageIdx) ? m.trainIdx : m.queryIdx;

                    // Tìm điểm 3D từ view đã đăng ký
                    const std::vector<Point3D>& existingPoints = triangulator.getPoints();
                    for (const auto& point : existingPoints) {
                        auto it = point.observations.find(regViewIdx);
                        if (it != point.observations.end() && it->second == keypointIdx2) {
                            points2D.push_back(images[imageIdx].getKeypoints()[keypointIdx1].pt);
                            points3D.push_back(point.position);
                            break;
                        }
                    }
                }
            }

            // Nếu không có đủ điểm để estimate pose, bỏ qua view này
            if (points2D.size() < 4) {
                continue;
            }

            // Estimate pose cho view mới
            if (poseEstimator.estimatePoseFromPoints(imageIdx, points2D, points3D)) {
                // Cập nhật lại cameraPoses
                const auto& existingPoses = poseEstimator.getCameraPoses();
                auto allCameraPoses = poseEstimator.getAllCameraPoses();
                
                for (size_t i = 0; i < allCameraPoses.size(); ++i) {
                    for (const auto& [index, pose] : existingPoses) {
                        // So sánh giá trị của camera pose
                        if (pose.rotation.at<double>(0,0) == allCameraPoses[i].rotation.at<double>(0,0) &&
                            pose.translation.at<double>(0,0) == allCameraPoses[i].translation.at<double>(0,0)) {
                            cameraPoses[index] = allCameraPoses[i];
                            break;
                        }
                    }
                }
            
                // Triangulate điểm mới
                size_t newPointsCount = triangulator.triangulateNewView(
                    images, imageIdx, matches, 
                    cameraPoses, 
                    registeredViews
                );
            
                std::cout << "Added view " << imageIdx 
                          << " with " << newPointsCount << " new points" << std::endl;
            
                registeredViews.push_back(imageIdx);
            }
        }

        // Lưu kết quả
        savePoses("camera_poses.txt");
        savePointCloud("point_cloud.ply");

        return true;
    }

    const ImageProcessing::MatchData* SfMReconstructor::findMatchBetweenImages(
        size_t imageIdx1, size_t imageIdx2) const {
        for (const auto& match : matches) {
            if ((match.imageIdx1 == imageIdx1 && match.imageIdx2 == imageIdx2) ||
                (match.imageIdx1 == imageIdx2 && match.imageIdx2 == imageIdx1)) {
                return &match;
            }
        }
        return nullptr;
    }

    bool SfMReconstructor::savePoses(const std::string& filename) const {
        return poseEstimator.savePoses(filename);
    }

    bool SfMReconstructor::savePointCloud(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open point cloud file: " << filename << std::endl;
            return false;
        }

        const std::vector<Point3D>& points = triangulator.getPoints();

        // Ghi header PLY
        file << "ply" << std::endl;
        file << "format ascii 1.0" << std::endl;
        file << "element vertex " << points.size() << std::endl;
        file << "property float x" << std::endl;
        file << "property float y" << std::endl;
        file << "property float z" << std::endl;
        file << "property uchar red" << std::endl;
        file << "property uchar green" << std::endl;
        file << "property uchar blue" << std::endl;
        file << "end_header" << std::endl;

        // Ghi các điểm
        for (const auto& point : points) {
            file << point.position.x << " " 
                << point.position.y << " " 
                << point.position.z << " "
                << static_cast<int>(point.color[2]) << " "  // OpenCV sử dụng BGR
                << static_cast<int>(point.color[1]) << " " 
                << static_cast<int>(point.color[0]) << std::endl;
        }

        file.close();
        std::cout << "Saved point cloud to " << filename << std::endl;
        return true;
    }

}
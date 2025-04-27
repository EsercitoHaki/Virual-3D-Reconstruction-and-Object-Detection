#include "IglReconstruction.h"
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <Eigen/Geometry>
#include <omp.h>

namespace Surface {
    IglReconstruction::IglReconstruction(ReconstructionMethod method) 
        : method(method) {
    }
    
    Mesh IglReconstruction::reconstruct(const std::vector<SFM::Point3D>& points) {
        switch (method) {
            case MARCHING_CUBES:
                return reconstructPointCloud(points);
            case BALL_PIVOTING:
            case ADVANCING_FRONT:
            default:
                return reconstructPointCloud(points);
        }
    }
    
    Mesh IglReconstruction::reconstructPointCloud(const std::vector<SFM::Point3D>& points) {
        std::cout << "Starting point cloud surface reconstruction..." << std::endl;
        
        if (points.empty()) {
            std::cerr << "Empty point cloud!" << std::endl;
            return Mesh();
        }
        
        std::cout << "Point cloud has " << points.size() << " points" << std::endl;
        
        Eigen::Vector3d min_corner(std::numeric_limits<double>::max(), 
                                    std::numeric_limits<double>::max(), 
                                    std::numeric_limits<double>::max());
        Eigen::Vector3d max_corner(std::numeric_limits<double>::lowest(), 
                                    std::numeric_limits<double>::lowest(), 
                                    std::numeric_limits<double>::lowest());
        
        std::vector<Eigen::Vector3d> eigenPoints;
        eigenPoints.reserve(points.size());
        
        for (const auto& pt : points) {
            Eigen::Vector3d point(pt.position.x, pt.position.y, pt.position.z);
            eigenPoints.push_back(point);
            
            for (int j = 0; j < 3; ++j) {
                min_corner(j) = std::min(min_corner(j), point(j));
                max_corner(j) = std::max(max_corner(j), point(j));
            }
        }
        
        Eigen::Vector3d centroid = (min_corner + max_corner) / 2.0;
        double radius = (max_corner - min_corner).norm() / 2.0;
        
        Mesh mesh;
        
        std::vector<size_t> vertexIndices;
        vertexIndices.reserve(points.size());
        
        for (size_t i = 0; i < points.size(); ++i) {
            Vertex vertex;
            vertex.position = cv::Point3f(
                eigenPoints[i](0), 
                eigenPoints[i](1), 
                eigenPoints[i](2)
            );
            
            const auto& originalPt = points[i];
            vertex.color = cv::Vec3b(
                originalPt.color[2],
                originalPt.color[1],
                originalPt.color[0]
            );
            
            mesh.addVertex(vertex);
            vertexIndices.push_back(i);
        }
        
        #pragma omp parallel
        {
            std::vector<Triangle> localTriangles;
            
            #pragma omp for
            for (size_t i = 0; i < vertexIndices.size(); ++i) {
                std::vector<size_t> neighbors;
                for (size_t j = 0; j < vertexIndices.size(); ++j) {
                    if (i == j) continue;
                    
                    double dist = (eigenPoints[i] - eigenPoints[j]).norm();
                    if (dist <= radius * 0.5) {
                        neighbors.push_back(j);
                    }
                    
                    if (neighbors.size() >= 10) break;
                }
                
                for (size_t k = 0; k + 1 < neighbors.size(); ++k) {
                    Triangle triangle;
                    triangle.vertices[0] = i;
                    triangle.vertices[1] = neighbors[k];
                    triangle.vertices[2] = neighbors[k+1];
                    
                    localTriangles.push_back(triangle);
                }
            }
            
            #pragma omp critical
            {
                for (const auto& triangle : localTriangles) {
                    mesh.addTriangle(triangle);
                }
            }
        }
        
        return mesh;
    }
}
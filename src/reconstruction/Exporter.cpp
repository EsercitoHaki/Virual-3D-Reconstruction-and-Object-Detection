#include "Exporter.h"
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

namespace Reconstruction {
    Exporter::Exporter(const std::string& outputDir) : outputDir(outputDir) {
        fs::create_directories(outputDir);
        outputFile = outputDir + "/matches.txt";
    }

    std::string Exporter::getFullPath(const std::string& filename) const {
        return (fs::path(outputDir) / filename).string();
    }

    bool Exporter::exportPointCloudToPLY(
        const std::vector<SFM::Point3D>& points, 
        const std::string& filename) 
    {
        std::string fullPath = getFullPath(filename);
        std::ofstream plyFile(fullPath);
        
        if (!plyFile.is_open()) {
            std::cerr << "Không thể mở file để xuất point cloud: " << fullPath << std::endl;
            return false;
        }

        plyFile << "ply\n";
        plyFile << "format ascii 1.0\n";
        plyFile << "element vertex " << points.size() << "\n";
        plyFile << "property float x\n";
        plyFile << "property float y\n";
        plyFile << "property float z\n";
        plyFile << "property uchar red\n";
        plyFile << "property uchar green\n";
        plyFile << "property uchar blue\n";
        plyFile << "property list uchar int observations\n";
        plyFile << "end_header\n";

        for (const auto& point : points) {
            plyFile << point.position.x << " " 
                    << point.position.y << " " 
                    << point.position.z << " "
                    << static_cast<int>(point.color[2]) << " "
                    << static_cast<int>(point.color[1]) << " "
                    << static_cast<int>(point.color[0]) << " ";
            
            plyFile << point.observations.size() << " ";
            for (const auto& obs : point.observations) {
                plyFile << obs.first << " ";
            }
            plyFile << "\n";
        }

        plyFile.close();
        std::cout << "Đã xuất point cloud sang PLY: " << fullPath 
                << " (Tổng số điểm: " << points.size() << ")" << std::endl;
        return true;
    }

    bool Exporter::exportMeshToOBJ(const Surface::Mesh& mesh, const std::string& filename) {
        std::string fullPath = getFullPath(filename);
        return mesh.saveToOBJ(fullPath);
    }

    bool Exporter::exportMeshToPLY(const Surface::Mesh& mesh, const std::string& filename) {
        std::string fullPath = getFullPath(filename);
        return mesh.saveToPLY(fullPath);
    }

    bool Exporter::exportMeshToSTL(const Surface::Mesh& mesh, const std::string& filename) {
        std::string fullPath = getFullPath(filename);
        std::ofstream stlFile(fullPath);
        
        if (!stlFile.is_open()) {
            std::cerr << "Không thể mở file để xuất mesh: " << fullPath << std::endl;
            return false;
        }

        stlFile << "solid reconstructed_mesh\n";

        const auto& vertices = mesh.getVertices();
        const auto& triangles = mesh.getTriangles();

        for (const auto& triangle : triangles) {
            const auto& v1 = vertices[triangle.vertices[0]];
            const auto& v2 = vertices[triangle.vertices[1]];
            const auto& v3 = vertices[triangle.vertices[2]];

            cv::Point3f normal = v1.normal;

            stlFile << "  facet normal " 
                    << normal.x << " " 
                    << normal.y << " " 
                    << normal.z << "\n";
            stlFile << "    outer loop\n";
            stlFile << "      vertex " 
                    << v1.position.x << " " 
                    << v1.position.y << " " 
                    << v1.position.z << "\n";
            stlFile << "      vertex " 
                    << v2.position.x << " " 
                    << v2.position.y << " " 
                    << v2.position.z << "\n";
            stlFile << "      vertex " 
                    << v3.position.x << " " 
                    << v3.position.y << " " 
                    << v3.position.z << "\n";
            stlFile << "    endloop\n";
            stlFile << "  endfacet\n";
        }

        stlFile << "endsolid reconstructed_mesh\n";

        stlFile.close();
        std::cout << "Đã xuất mesh sang STL: " << fullPath << std::endl;
        return true;
    }

    std::string Exporter::getOutputFilePath() const {
        return outputFile;
    }
}
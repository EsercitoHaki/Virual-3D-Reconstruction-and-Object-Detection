// src/surface/Mesh.cpp
#include "Mesh.h"
#include <fstream>
#include <iostream>

namespace Surface {
    void Mesh::addVertex(const Vertex& vertex) {
        vertices.push_back(vertex);
    }
    
    void Mesh::addTriangle(const Triangle& triangle) {
        triangles.push_back(triangle);
    }
    
    bool Mesh::saveToOBJ(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return false;
        }
        
        file << "# Mesh exported from SfM project\n";
        file << "# Vertices: " << vertices.size() << "\n";
        file << "# Faces: " << triangles.size() << "\n\n";
        
        for (const auto& vertex : vertices) {
            file << "v " << vertex.position.x << " " 
                 << vertex.position.y << " " 
                 << vertex.position.z << " "
                 << static_cast<float>(vertex.color[2])/255.0f << " " 
                 << static_cast<float>(vertex.color[1])/255.0f << " " 
                 << static_cast<float>(vertex.color[0])/255.0f << "\n";
        }
        
        for (const auto& vertex : vertices) {
            file << "vn " << vertex.normal.x << " " 
                 << vertex.normal.y << " " 
                 << vertex.normal.z << "\n";
        }
        
        for (const auto& triangle : triangles) {
            file << "f " 
                 << (triangle.vertices[0] + 1) << "//" << (triangle.vertices[0] + 1) << " "
                 << (triangle.vertices[1] + 1) << "//" << (triangle.vertices[1] + 1) << " "
                 << (triangle.vertices[2] + 1) << "//" << (triangle.vertices[2] + 1) << "\n";
        }
        
        file.close();
        return true;
    }
    
    bool Mesh::saveToPLY(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return false;
        }
        
        file << "ply\n";
        file << "format ascii 1.0\n";
        file << "element vertex " << vertices.size() << "\n";
        file << "property float x\n";
        file << "property float y\n";
        file << "property float z\n";
        file << "property float nx\n";
        file << "property float ny\n";
        file << "property float nz\n";
        file << "property uchar red\n";
        file << "property uchar green\n";
        file << "property uchar blue\n";
        file << "element face " << triangles.size() << "\n";
        file << "property list uchar int vertex_indices\n";
        file << "end_header\n";
        
        for (const auto& vertex : vertices) {
            file << vertex.position.x << " " 
                 << vertex.position.y << " " 
                 << vertex.position.z << " "
                 << vertex.normal.x << " " 
                 << vertex.normal.y << " " 
                 << vertex.normal.z << " "
                 << static_cast<int>(vertex.color[2]) << " " 
                 << static_cast<int>(vertex.color[1]) << " " 
                 << static_cast<int>(vertex.color[0]) << "\n";
        }

        for (const auto& triangle : triangles) {
            file << "3 " << triangle.vertices[0] << " " 
                 << triangle.vertices[1] << " " 
                 << triangle.vertices[2] << "\n";
        }
        
        file.close();
        return true;
    }
}
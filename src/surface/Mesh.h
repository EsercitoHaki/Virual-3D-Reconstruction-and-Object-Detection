#pragma once

#include <vector>
#include <array>
#include <string>
#include <opencv2/core.hpp>

namespace Surface {
    struct Vertex {
        cv::Point3f position;
        cv::Point3f normal;
        cv::Vec3b color;
    };

    struct Triangle {
        std::array<size_t, 3> vertices;
    };

    class Mesh {
    public:
        Mesh() = default;
        
        void addVertex(const Vertex& vertex);
        void addTriangle(const Triangle& triangle);
        
        bool saveToOBJ(const std::string& filename) const;
        bool saveToPLY(const std::string& filename) const;
        
        const std::vector<Vertex>& getVertices() const { return vertices; }
        const std::vector<Triangle>& getTriangles() const { return triangles; }
        
    private:
        std::vector<Vertex> vertices;
        std::vector<Triangle> triangles;
    };
}
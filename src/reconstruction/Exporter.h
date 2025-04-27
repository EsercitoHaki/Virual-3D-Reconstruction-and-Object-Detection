#pragma once

#include "../analysis/ImageData.h"
#include "../analysis/FeatureMatcher.h"
#include "../surface/Mesh.h"
#include "../sfm/Triangulator.h" 
#include <string>
#include <vector>

namespace Reconstruction {

    class Exporter {
        public:
            explicit Exporter(const std::string& outputDir);
            
            bool exportMatchesForReconstruction(
                const std::vector<ImageProcessing::ImageData>& images,
                const std::vector<ImageProcessing::MatchData>& matches,
                size_t minMatches = 20);

            bool exportPointCloudToPLY(
                const std::vector<SFM::Point3D>& points, 
                const std::string& filename = "point_cloud.ply"
            );
                
            bool exportMeshToOBJ(const Surface::Mesh& mesh, const std::string& filename = "reconstructed_surface.obj");
            bool exportMeshToPLY(const Surface::Mesh& mesh, const std::string& filename = "reconstructed_surface.ply");
            bool exportMeshToSTL(const Surface::Mesh& mesh, const std::string& filename = "reconstructed_surface.stl");
            
            std::string getOutputFilePath() const;

        private:
            void exportCameraPositionsPLY(
                const std::vector<ImageProcessing::ImageData>& images,
                const std::vector<ImageProcessing::MatchData>& matches,
                size_t minMatches);
            
            std::string getFullPath(const std::string& filename) const;
            
            std::string outputDir;
            std::string outputFile;
    };

}
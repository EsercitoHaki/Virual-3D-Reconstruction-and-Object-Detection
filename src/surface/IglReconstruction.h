#pragma once

#include "Mesh.h"
#include "../sfm/Triangulator.h"
#include <Eigen/Dense>

namespace Surface {
    class IglReconstruction {
    public:
        enum ReconstructionMethod {
            MARCHING_CUBES,
            BALL_PIVOTING,
            ADVANCING_FRONT
        };
        
        IglReconstruction(ReconstructionMethod method = MARCHING_CUBES);
        
        void setGridResolution(int resolution) { gridResolution = resolution; }
        void setIsoLevel(double level) { isoLevel = level; }
        
        Mesh reconstruct(const std::vector<SFM::Point3D>& points);
        
    private:
        ReconstructionMethod method;
        
        int gridResolution = 50;
        double isoLevel = 0.01;
        
        Mesh reconstructPointCloud(const std::vector<SFM::Point3D>& points);
    };
}
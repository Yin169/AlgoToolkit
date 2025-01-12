#ifndef VISUAL_HPP
#define VISUAL_HPP

#include <fstream>
#include <string>
#include <iomanip>
#include "../Mesh/MeshObj.hpp"

template<typename T, size_t D>
class Visual {
public:
    static void writeVTK(const MeshObj<T,D>& mesh, const std::string& filename) {
        std::ofstream file(filename + ".vtk");
        file << std::scientific << std::setprecision(6);
        writeHeader(file);
        writePoints(file, mesh);
        writeCells(file, mesh);
        writeData(file, mesh);
        file.close();
    }

private:
    static void writeHeader(std::ofstream& file) {
        file << "# vtk DataFile Version 2.0\n";
        file << "LBM Mesh Visualization\n";
        file << "ASCII\n";
        file << "DATASET UNSTRUCTURED_GRID\n";
    }

    static void writePoints(std::ofstream& file, const MeshObj<T,D>& mesh) {
        const auto& nodes = mesh.getNodes();
        file << "POINTS " << nodes.size() << " double\n";
        
        for(const auto& node : nodes) {
            for(size_t d = 0; d < D; d++) {
                file << node.position[d] << " ";
            }
            if(D == 2) file << "0.0 ";
            file << "\n";
        }
    }

    static void writeCells(std::ofstream& file, const MeshObj<T,D>& mesh) {
        const auto& nodes = mesh.getNodes();
        std::vector<size_t> activeNodes;
        
        for(size_t i = 0; i < nodes.size(); i++) {
            if(nodes[i].isActive) activeNodes.push_back(i);
        }

        // VTK cell types: 10=tetra, 5=triangle
        const int cellType = (D == 2) ? 5 : 10;
        const size_t pointsPerCell = (D == 2) ? 3 : 4;
        
        file << "\nCELLS " << activeNodes.size() << " " 
             << activeNodes.size() * (pointsPerCell + 1) << "\n";
        
        for(size_t idx : activeNodes) {
            file << pointsPerCell << " " << idx;
            for(size_t i = 1; i <= D; i++) {
                if(i < nodes[idx].neighbors.size()) {
                    file << " " << nodes[idx].neighbors[i];
                }
            }
            file << "\n";
        }

        file << "\nCELL_TYPES " << activeNodes.size() << "\n";
        for(size_t i = 0; i < activeNodes.size(); i++) {
            file << cellType << "\n";
        }
    }

    static void writeData(std::ofstream& file, const MeshObj<T,D>& mesh) {
        const auto& nodes = mesh.getNodes();
        
        file << "\nPOINT_DATA " << nodes.size() << "\n";
        
        // Level data
        file << "SCALARS Level double 1\n";
        file << "LOOKUP_TABLE default\n";
        for(const auto& node : nodes) {
            file << static_cast<double>(node.level) << "\n";
        }

        // Active status
        file << "\nSCALARS Active double 1\n";
        file << "LOOKUP_TABLE default\n";
        for(const auto& node : nodes) {
            file << (node.isActive ? 1.0 : 0.0) << "\n";
        }

        // Velocity distributions
        file << "\nVECTORS Velocity double\n";
        for(const auto& node : nodes) {
            double vx = 0.0, vy = 0.0, vz = 0.0;
            for(size_t i = 0; i < node.distributions.size(); i++) {
                if(D == 2) {
                    vx += node.distributions[i] * LatticeTraits<2>::directions[i][0];
                    vy += node.distributions[i] * LatticeTraits<2>::directions[i][1];
                } else {
                    vx += node.distributions[i] * LatticeTraits<3>::directions[i][0];
                    vy += node.distributions[i] * LatticeTraits<3>::directions[i][1];
                    vz += node.distributions[i] * LatticeTraits<3>::directions[i][2];
                }
            }
            file << vx << " " << vy << " " << (D == 2 ? 0.0 : vz) << "\n";
        }
    }
};

#endif // VISUAL_HPP
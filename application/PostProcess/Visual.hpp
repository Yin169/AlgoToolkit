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
        size_t numActiveCells = 0;
        
        // Count active nodes
        for(const auto& node : nodes) {
            if(node.isActive) numActiveCells++;
        }
        
        // Write cell connectivity
        file << "CELLS " << numActiveCells << " " 
             << numActiveCells * (D == 2 ? 4 : 5) << "\n";
        
        for(size_t i = 0; i < nodes.size(); i++) {
            if(!nodes[i].isActive) continue;
            
            if(D == 2) {
                file << "3 " << i << " ";
                if(nodes[i].neighbors.size() >= 2) {
                    file << nodes[i].neighbors[0] << " " 
                         << nodes[i].neighbors[1];
                } else {
                    file << i << " " << i;  // Degenerate case
                }
            } else {
                file << "4 " << i << " ";
                if(nodes[i].neighbors.size() >= 3) {
                    file << nodes[i].neighbors[0] << " "
                         << nodes[i].neighbors[1] << " "
                         << nodes[i].neighbors[2];
                } else {
                    file << i << " " << i << " " << i;  // Degenerate case
                }
            }
            file << "\n";
        }
        
        // Write cell types
        file << "CELL_TYPES " << numActiveCells << "\n";
        for(const auto& node : nodes) {
            if(!node.isActive) continue;
            file << (D == 2 ? 5 : 10) << "\n";  // 5=triangle, 10=tetra
        }
    }

    static void writeData(std::ofstream& file, const MeshObj<T,D>& mesh) {
        const auto& nodes = mesh.getNodes();
        size_t numActive = 0;
        for(const auto& node : nodes) {
            if(node.isActive) numActive++;
        }
        
        file << "\nPOINT_DATA " << nodes.size() << "\n";
        writeScalarData(file, "Density", nodes);
        writeVectorData(file, "Velocity", nodes);
    }

    static void writeScalarData(std::ofstream& file, const std::string& name, const std::vector<LBMNode<T,D>>& nodes) {
        file << "SCALARS " << name << " double 1\n";
        file << "LOOKUP_TABLE default\n";
        for(const auto& node : nodes) {
            T density = std::accumulate(node.distributions.begin(), node.distributions.end(), T(0));
            file << density << "\n";
        }
    }

    static void writeVectorData(std::ofstream& file, const std::string& name, const std::vector<LBMNode<T,D>>& nodes) {
        file << "\nVECTORS " << name << " double\n";
        for(const auto& node : nodes) {
            std::array<T,D> velocity = {};
            T density = std::accumulate(node.distributions.begin(), node.distributions.end(), T(0));

            for(size_t i = 0; i < LatticeTraits<D>::Q; i++) {
                for(size_t d = 0; d < D; d++) {
                    velocity[d] += node.distributions[i] * LatticeTraits<D>::directions[i][d];
                }
            }

            for(size_t d = 0; d < D; d++) {
                velocity[d] /= density;
                file << velocity[d] << " ";
            }
            if(D == 2) file << "0.0";
            file << "\n";
        }
    }
};

#endif // VISUAL_HPP
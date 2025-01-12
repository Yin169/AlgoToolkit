#ifndef MESH_OBJ_HPP
#define MESH_OBJ_HPP

#include "VectorObj.hpp"
#include <vector>
#include <array>
#include <numeric>
#include <functional>

// Lattice model traits
template<size_t D> struct LatticeTraits;

template<>
struct LatticeTraits<2> {
    static constexpr size_t Q = 9;
    static constexpr std::array<std::array<int, 2>, 9> directions = {{
        {0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1},
        {1, 1}, {-1, 1}, {-1, -1}, {1, -1}
    }};
};

template<>
struct LatticeTraits<3> {
    static constexpr size_t Q = 19;
    static constexpr std::array<std::array<int, 3>, 19> directions = {{
        {0,0,0},
        {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1},
        {1,1,0}, {-1,-1,0}, {1,-1,0}, {-1,1,0},
        {1,0,1}, {-1,0,-1}, {1,0,-1}, {-1,0,1},
        {0,1,1}, {0,-1,-1}, {0,1,-1}, {0,-1,1}
    }};
};

template<typename T, size_t D>
struct LBMNode {
    std::array<T, D> position;
    std::array<T, LatticeTraits<D>::Q> distributions;
    size_t level;
    bool isActive;
    std::vector<size_t> neighbors;
};

template<typename T, size_t D>
class MeshObj {
private:
    std::vector<LBMNode<T,D>> nodes;
    std::array<size_t, D> dimensions;
    T dx;
    T refinementThreshold;
    size_t maxLevel;
    
public:
    MeshObj(const std::array<size_t, D>& dims, T dx_, T threshold = 0.1) 
        : dimensions(dims), dx(dx_), refinementThreshold(threshold), maxLevel(3) {
        initializeGrid();
    }

    void initializeGrid() {
        size_t totalNodes = std::accumulate(dimensions.begin(), dimensions.end(), 
                                          1, std::multiplies<size_t>());
        nodes.resize(totalNodes);
        
        std::array<size_t, D> indices{};
        initializeGridRecursive(indices, 0);
    }

	const std::vector<LBMNode<T,D>>& getNodes() const {
		return nodes;
	}

	const std::vector<size_t> getNeighbors(size_t idx) const {
		return nodes[idx].neighbors;
	}

    void adaptGrid(const VectorObj<T>& macroscopic) {
        std::vector<size_t> refineList;
        std::vector<size_t> coarsenList;

        // Mark cells for refinement/coarsening
        for(size_t i = 0; i < nodes.size(); i++) {
            if(!nodes[i].isActive) continue;

            if(needsRefinement(macroscopic, i)) {
                refineList.push_back(i);
            } else if(needsCoarsening(macroscopic, i)) {
                coarsenList.push_back(i);
            }
        }

        // Process refinement
        for(auto idx : refineList) {
            refineCell(idx);
        }

        // Process coarsening
        for(auto idx : coarsenList) {
            coarsenCell(idx);
        }

        // Update connectivity
        updateConnectivity();
    }

private:
    void initializeGridRecursive(std::array<size_t, D>& indices, size_t depth) {
        if(depth == D) {
            size_t idx = flattenIndex(indices);
            nodes[idx].position = calculatePosition(indices);
            nodes[idx].level = 0;
            nodes[idx].isActive = true;
            setNeighbors(idx);
            return;
        }

        for(size_t i = 0; i < dimensions[depth]; i++) {
            indices[depth] = i;
            initializeGridRecursive(indices, depth + 1);
        }
    }

    size_t flattenIndex(const std::array<size_t, D>& indices) {
        size_t idx = 0;
        size_t stride = 1;
        for(size_t d = 0; d < D; d++) {
            idx += indices[d] * stride;
            stride *= dimensions[d];
        }
        return idx;
    }

    std::array<T, D> calculatePosition(const std::array<size_t, D>& indices) {
        std::array<T, D> pos;
        for(size_t d = 0; d < D; d++) {
            pos[d] = indices[d] * dx;
        }
        return pos;
    }

    void setNeighbors(size_t idx) {
        std::array<size_t, D> indices = unflattenIndex(idx);
        nodes[idx].neighbors.clear();
        
        for(const auto& dir : LatticeTraits<D>::directions) {
            bool valid = true;
            std::array<size_t, D> neighborIndices;
            
            for(size_t d = 0; d < D; d++) {
                int ni = static_cast<int>(indices[d]) + dir[d];
                if(ni < 0 || ni >= static_cast<int>(dimensions[d])) {
                    valid = false;
                    break;
                }
                neighborIndices[d] = ni;
            }
            
            if(valid) {
                nodes[idx].neighbors.push_back(flattenIndex(neighborIndices));
            }
        }
    }

    std::array<size_t, D> unflattenIndex(size_t idx) {
        std::array<size_t, D> indices;
        for(size_t d = 0; d < D; d++) {
            size_t stride = 1;
            for(size_t k = 0; k < d; k++) {
                stride *= dimensions[k];
            }
            indices[d] = (idx / stride) % dimensions[d];
        }
        return indices;
    }

    bool needsRefinement(const VectorObj<T>& macro, size_t idx) {
        if(nodes[idx].level >= maxLevel) return false;
        return computeGradient(macro, idx) > refinementThreshold;
    }

    bool needsCoarsening(const VectorObj<T>& macro, size_t idx) {
        if(nodes[idx].level == 0) return false;
        return computeGradient(macro, idx) < refinementThreshold/2;
    }

    T computeGradient(const VectorObj<T>& macro, size_t idx) {
        T maxGrad = 0;
        for(size_t n : nodes[idx].neighbors) {
            T grad = std::abs(macro[idx] - macro[n]) / dx;
            maxGrad = std::max(maxGrad, grad);
        }
        return maxGrad;
    }

    void refineCell(size_t idx) {
        if(nodes[idx].level >= maxLevel) return;

        // Create new nodes at half spacing
        T newDx = dx / (1 << (nodes[idx].level + 1));
        std::array<T, D> basePos = nodes[idx].position;

        // Add 4 new nodes (2x2 refinement)
        for(int j = 0; j < 2; j++) {
            for(int i = 0; i < 2; i++) {
                LBMNode<T, D> newNode;
                newNode.position = {
                    basePos[0] + i * newDx,
                    basePos[1] + j * newDx
                };
                newNode.level = nodes[idx].level + 1;
                newNode.isActive = true;
                nodes.push_back(newNode);
            }
        }

        nodes[idx].isActive = false;
    }

    void coarsenCell(size_t idx) {
        if(nodes[idx].level == 0) return;

        // Find all child nodes and deactivate them
        T cellDx = dx / (1 << nodes[idx].level);
        std::array<T, D> basePos = nodes[idx].position;

        for(size_t i = 0; i < nodes.size(); i++) {
            if(!nodes[i].isActive) continue;
            if(nodes[i].level != nodes[idx].level) continue;

            T dx = nodes[i].position[0] - basePos[0];
            T dy = nodes[i].position[1] - basePos[1];

            if(dx >= 0 && dx < cellDx && dy >= 0 && dy < cellDx) {
                nodes[i].isActive = false;
            }
        }

        // Reactivate parent node
        nodes[idx].isActive = true;
        nodes[idx].level--;
    }

    void updateConnectivity() {
        for(size_t i = 0; i < nodes.size(); i++) {
            if(nodes[i].isActive) {
                setNeighbors(i);
            }
        }
    }
};

#endif // MESH_OBJ_HPP
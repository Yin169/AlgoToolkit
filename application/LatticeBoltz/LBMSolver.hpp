#ifndef LBM_SOLVER_HPP
#define LBM_SOLVER_HPP

#include "../Mesh/MeshObj.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <unordered_map>

// Boundary types
enum class BoundaryType {
    NoSlip,
    VelocityInlet,
    PressureOutlet,
    IBM
};

// IBM node structure
template<typename T, size_t D>
struct IBMNode {
    std::array<T, D> position;
    std::array<T, D> velocity;
    std::array<T, D> force;
    std::vector<size_t> nearNodes;
    std::vector<T> weights;
};

template<typename T, size_t D>
class LBMSolver {
private:
    MeshObj<T,D>& mesh;
    T omega;        // Relaxation parameter
    T cs2;         // Speed of sound squared
    T deltaX;      // Grid spacing
    T deltaT;      // Time step
    std::array<T, LatticeTraits<D>::Q> weights;
    std::vector<std::pair<size_t, BoundaryType>> boundaries;
    std::vector<IBMNode<T,D>> ibmNodes;
    std::unordered_map<size_t, std::array<T,D>> inletVelocities;
    std::unordered_map<size_t, T> outletPressures;

    void initializeWeights() {
        if constexpr (D == 2) {
            weights = {4.0/9.0,  
                      1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                      1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
        } else {
            weights = {1.0/3.0,
                      1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
                      1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
                      1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
                      1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
        }
    }

    void applyBoundaryConditions() {
        for(const auto& [nodeIdx, type] : boundaries) {
            switch(type) {
                case BoundaryType::NoSlip:
                    applyBounceBack(nodeIdx);
                    break;
                case BoundaryType::VelocityInlet:
                    applyZouHeVelocity(nodeIdx, inletVelocities[nodeIdx]);
                    break;
                case BoundaryType::PressureOutlet:
                    applyZouHePressure(nodeIdx, outletPressures[nodeIdx]);
                    break;
            }
        }
    }

    void applyBounceBack(size_t idx) {
        auto& node = mesh.getNodes()[idx];
        for(size_t i = 1; i < LatticeTraits<D>::Q; i++) {
            std::swap(node.distributions[i], 
                     node.distributions[LatticeTraits<D>::Q - i]);
        }
    }

    void applyZouHeVelocity(size_t idx, const std::array<T,D>& velocity) {
        auto& node = mesh.getNodes()[idx];
        T rho = computeDensity(node.distributions);
        auto feq = computeEquilibrium(rho, velocity);
        node.distributions = feq;
    }

    void applyZouHePressure(size_t idx, T pressure) {
        auto& node = mesh.getNodes()[idx];
        auto velocity = computeVelocity(node.distributions, pressure);
        auto feq = computeEquilibrium(pressure, velocity);
        node.distributions = feq;
    }

    void updateIBMForces() {
        for(auto& ibmNode : ibmNodes) {
            // Interpolate fluid velocity at IBM position
            std::array<T,D> uf = {};
            T totalWeight = 0;
            
            for(size_t i = 0; i < ibmNode.nearNodes.size(); i++) {
                auto& node = mesh.getNodes()[ibmNode.nearNodes[i]];
                auto u = computeVelocity(node.distributions, 
                                       computeDensity(node.distributions));
                
                for(size_t d = 0; d < D; d++) {
                    uf[d] += u[d] * ibmNode.weights[i];
                }
                totalWeight += ibmNode.weights[i];
            }
            
            // Normalize and calculate force
            for(size_t d = 0; d < D; d++) {
                uf[d] /= totalWeight;
                ibmNode.force[d] = (ibmNode.velocity[d] - uf[d])/deltaT;
            }
            
            // Distribute force to fluid nodes
            for(size_t i = 0; i < ibmNode.nearNodes.size(); i++) {
                auto& node = mesh.getNodes()[ibmNode.nearNodes[i]];
                for(size_t j = 0; j < LatticeTraits<D>::Q; j++) {
                    T force = 0;
                    for(size_t d = 0; d < D; d++) {
                        force += LatticeTraits<D>::directions[j][d] * 
                                ibmNode.force[d];
                    }
                    node.distributions[j] += weights[j] * force * 
                                           ibmNode.weights[i]/totalWeight;
                }
            }
        }
    }

    void collision() {
        auto& nodes = mesh.getNodes();
        #pragma omp parallel for
        for(size_t i = 0; i < nodes.size(); i++) {
            if(!nodes[i].isActive || nodes[i].isBoundary) continue;
            
            T rho = computeDensity(nodes[i].distributions);
            auto u = computeVelocity(nodes[i].distributions, rho);
            auto feq = computeEquilibrium(rho, u);
            
            for(size_t k = 0; k < LatticeTraits<D>::Q; k++) {
                nodes[i].distributions[k] += 
                    omega * (feq[k] - nodes[i].distributions[k]);
            }
        }
    }

    void streaming() {
        auto& nodes = mesh.getNodes();
        std::vector<std::array<T, LatticeTraits<D>::Q>> temp(nodes.size());
        
        #pragma omp parallel for
        for(size_t i = 0; i < nodes.size(); i++) {
            if(!nodes[i].isActive) continue;
            
            for(size_t k = 0; k < nodes[i].neighbors.size(); k++) {
                size_t neighbor = nodes[i].neighbors[k];
                if(k < LatticeTraits<D>::Q) {
                    temp[neighbor][k] = nodes[i].distributions[k];
                }
            }
        }
        
        for(size_t i = 0; i < nodes.size(); i++) {
            if(nodes[i].isActive) {
                nodes[i].distributions = temp[i];
            }
        }
    }

    T computeDensity(const std::array<T, LatticeTraits<D>::Q>& f) {
        return std::accumulate(f.begin(), f.end(), T(0));
    }

    std::array<T,D> computeVelocity(const std::array<T, LatticeTraits<D>::Q>& f, 
                                   T rho) {
        std::array<T,D> velocity = {};
        for(size_t i = 0; i < LatticeTraits<D>::Q; i++) {
            for(size_t d = 0; d < D; d++) {
                velocity[d] += f[i] * LatticeTraits<D>::directions[i][d];
            }
        }
        for(size_t d = 0; d < D; d++) velocity[d] /= rho;
        return velocity;
    }

    std::array<T, LatticeTraits<D>::Q> computeEquilibrium(T rho, 
                                                         const std::array<T,D>& u) {
        std::array<T, LatticeTraits<D>::Q> feq;
        T usqr = 0;
        for(size_t d = 0; d < D; d++) usqr += u[d] * u[d];
        
        for(size_t i = 0; i < LatticeTraits<D>::Q; i++) {
            T cu = 0;
            for(size_t d = 0; d < D; d++) {
                cu += LatticeTraits<D>::directions[i][d] * u[d];
            }
            feq[i] = weights[i] * rho * (1 + 3*cu + 4.5*cu*cu - 1.5*usqr);
        }
        return feq;
    }

public:
    LBMSolver(MeshObj<T,D>& mesh_, T viscosity, T dx = 1.0, T dt = 1.0) 
        : mesh(mesh_), deltaX(dx), deltaT(dt) {
        cs2 = 1.0/3.0;
        T tau = viscosity/cs2 + 0.5;
        omega = 1.0/tau;
        initializeWeights();
    }

    void initialize(T rho0 = 1.0, const std::array<T,D>& u0 = {}) {
        auto& nodes = mesh.getNodes();
        for(auto& node : nodes) {
            if(!node.isActive) continue;
            auto feq = computeEquilibrium(rho0, u0);
            node.distributions = feq;
        }
    }

    void collideAndStream() {
        applyBoundaryConditions();
        updateIBMForces();
        collision();
        streaming();
        applyBoundaryConditions();
    }

    void setBoundary(size_t nodeIdx, BoundaryType type) {
        boundaries.emplace_back(nodeIdx, type);
        mesh.setBoundaryNode(nodeIdx);
    }

    void setInletVelocity(size_t nodeIdx, const std::array<T,D>& velocity) {
        inletVelocities[nodeIdx] = velocity;
    }

    void setOutletPressure(size_t nodeIdx, T pressure) {
        outletPressures[nodeIdx] = pressure;
    }

    void addIBMNode(const IBMNode<T,D>& node) {
        ibmNodes.push_back(node);
    }

    const std::vector<IBMNode<T,D>>& getIBMNodes() const {
        return ibmNodes;
    }
};

#endif // LBM_SOLVER_HPP
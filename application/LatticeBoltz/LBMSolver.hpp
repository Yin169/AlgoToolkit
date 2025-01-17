#ifndef LBM_SOLVER_HPP
#define LBM_SOLVER_HPP

#include "../Mesh/MeshObj.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <unordered_map>

enum class BoundaryType {
    NoSlip,
    VelocityInlet,
    PressureOutlet,
    IBM
};

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

    void collision() {
        auto& nodes = mesh.getNodes();
        // #pragma omp parallel for
        for(size_t i = 0; i < nodes.size(); i++) {
            if(!nodes[i].isActive) continue;
            
            T rho = computeDensity(nodes[i].distributions);
            auto u = computeVelocity(nodes[i].distributions, rho);
            auto feq = computeEquilibrium(rho, u);
            
            for(size_t k = 0; k < LatticeTraits<D>::Q; k++) {
                nodes[i].distributions[k] += omega * (feq[k] - nodes[i].distributions[k]);
                // std::cout << nodes[i].distributions[k] << " " << feq[k] << " " << omega << std::endl;
            }
        }
    }

    void streaming() {
        auto& nodes = mesh.getNodes();
        std::vector<std::array<T, LatticeTraits<D>::Q>> temp(nodes.size());
        
        // First store all distributions
        for(size_t i = 0; i < nodes.size(); i++) {
            temp[i] = nodes[i].distributions;
        }
        
        // Then stream
        for(size_t i = 0; i < nodes.size(); i++) {
            if(!nodes[i].isActive) continue;
            
            for(size_t k = 0; k < LatticeTraits<D>::Q; k++) {
                size_t neighborIdx = mesh.getNeighborIndex(i, k);
                if(neighborIdx < nodes.size()) {
                    nodes[neighborIdx].distributions[k] = temp[i][k];
                }
            }
        }
    }

    std::array<T, LatticeTraits<D>::Q> computeEquilibrium(T rho, const std::array<T,D>& u) const {
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

    void applyBoundaryConditions() {
        for(const auto& [nodeIdx, type] : boundaries) {
            if (type ==  BoundaryType::NoSlip){
                    applyBounceBack(nodeIdx);
            } else if (type == BoundaryType::VelocityInlet){
                    applyZouHeVelocity(nodeIdx, inletVelocities[nodeIdx]);
            } else if (type == BoundaryType::PressureOutlet){
                    applyZouHePressure(nodeIdx, outletPressures[nodeIdx]);
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
        node.distributions = computeEquilibrium(rho, velocity);
    }

    void applyZouHePressure(size_t idx, T pressure) {
        auto& node = mesh.getNodes()[idx];
        auto velocity = computeVelocity(node.distributions, pressure);
        node.distributions = computeEquilibrium(pressure, velocity);
    }

public:
    LBMSolver(MeshObj<T,D>& mesh_, T viscosity, T dx = 1.0, T dt = 1.0) 
        : mesh(mesh_), deltaX(dx), deltaT(dt) {
        cs2 = 1.0/3.0;
        T tau = viscosity/cs2 + 4.0;
        omega = 1.0/tau;
        initializeWeights();
    }

    void initialize(T rho0 = 1.0, const std::array<T,D>& u0 = {}) {
        auto& nodes = mesh.getNodes();
        for(auto& node : nodes) {
            if(!node.isActive) continue;
            node.distributions = computeEquilibrium(rho0, u0);
        }
    }

    void collideAndStream() {
        applyBoundaryConditions();
        collision();
        streaming();
    }

    T computeDensity(const std::array<T, LatticeTraits<D>::Q>& f) const {
        return std::accumulate(f.begin(), f.end(), T(0));
    }

    std::array<T,D> computeVelocity(const std::array<T, LatticeTraits<D>::Q>& f, T rho) const {
        std::array<T,D> velocity = {};
        for(size_t i = 0; i < LatticeTraits<D>::Q; i++) {
            for(size_t d = 0; d < D; d++) {
                velocity[d] += f[i] * LatticeTraits<D>::directions[i][d];
            }
        }
        for(size_t d = 0; d < D; d++) velocity[d] /= rho;
        return velocity;
    }

    std::vector<IBMNode<T,D>> getIBMNodes () const {
        return ibmNodes;
    }
    
    void addIBMNode(const IBMNode<T,D>& node) {
        ibmNodes.push_back(node);
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
};

#endif // LBM_SOLVER_HPP
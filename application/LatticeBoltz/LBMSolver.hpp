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
            equilibrium(node.distributions, rho0, u0);
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
		for(auto idx : node.nearNodes) {
			mesh.setBoundaryNode(idx);
		}
	}
	
    const std::vector<IBMNode<T,D>>& getIBMNodes() const {
        return ibmNodes;
    }

    T deltaFunction(const std::array<T,D>& r) {
        T prod = 1.0;
        for(size_t d = 0; d < D; d++) {
            T q = std::abs(r[d])/deltaX;
            if(q > 2) return 0;
            prod *= (1 + std::cos(M_PI*q/2))/2;
        }
        return prod/std::pow(deltaX,D);
    }

    std::array<T,D> computeVelocity(const std::array<T,LatticeTraits<D>::Q>& f, T rho) {
        std::array<T,D> velocity = {};
        for(size_t i = 0; i < LatticeTraits<D>::Q; i++) {
            for(size_t d = 0; d < D; d++) {
                velocity[d] += f[i] * LatticeTraits<D>::directions[i][d];
            }
        }
        for(size_t d = 0; d < D; d++) velocity[d] /= rho;
        return velocity;
    }

private:
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
        equilibrium(node.distributions, rho, velocity);
    }

    void applyZouHePressure(size_t idx, T pressure) {
        auto& node = mesh.getNodes()[idx];
        auto velocity = computeVelocity(node.distributions, pressure);
        equilibrium(node.distributions, pressure, velocity);
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
            
            // Normalize interpolated velocity
            for(size_t d = 0; d < D; d++) {
                uf[d] /= totalWeight;
                // Calculate force
                ibmNode.force[d] = (ibmNode.velocity[d] - uf[d])/deltaT;
            }
            
            // Distribute force back to fluid nodes
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

    void initializeWeights() {
        if constexpr (D == 2) { // D2Q9
            weights = {4.0/9.0,  // rest
                      1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,  // orthogonal
                      1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0}; // diagonal
        } else { // D3Q19
            weights = {1.0/3.0,  // rest
                      1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,  // orthogonal
                      1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,  // diagonal plane xy
                      1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,  // diagonal plane xz
                      1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0}; // diagonal plane yz
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
                case BoundaryType::IBM:
                    // IBM forces are handled separately in updateIBMForces()
                    break;
            }
        }
    }

    void equilibrium(std::array<T, LatticeTraits<D>::Q>& feq, T rho, const std::array<T,D>& u) {
        T usqr = 0;
        for(size_t d = 0; d < D; d++) usqr += u[d] * u[d];

        for(size_t i = 0; i < LatticeTraits<D>::Q; i++) {
            T cu = 0;
            for(size_t d = 0; d < D; d++) {
                cu += LatticeTraits<D>::directions[i][d] * u[d];
            }
            feq[i] = weights[i] * rho * (1 + 3*cu + 4.5*cu*cu - 1.5*usqr);
        }
    }

    void collision() {
        auto& nodes = mesh.getNodes();
        #pragma omp parallel for
        for(size_t i = 0; i < nodes.size(); i++) {
            if(!nodes[i].isActive) continue;

            T rho = computeDensity(nodes[i].distributions);
            auto u = computeVelocity(nodes[i].distributions, rho);
            
            std::array<T, LatticeTraits<D>::Q> feq;
            equilibrium(feq, rho, u);

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
                temp[neighbor][k] = nodes[i].distributions[k];
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

};

#endif // LBM_SOLVER_HPP
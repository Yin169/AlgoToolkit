#ifndef LBM_SOLVER_HPP
#define LBM_SOLVER_HPP

#include "../Mesh/MeshObj.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <numeric>

enum class BoundaryType {
    NoSlip,
    VelocityInlet,
    PressureOutlet
};

template<typename T, size_t D>
struct Wall {
    std::array<T, D> normal;
    std::array<T, D> position;
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
    T omega;
    T cs2;
    T deltaX;
    T deltaT;
    T Cs;
    T minRho;
    T maxRho;
    T maxVelocity;
    std::array<T, LatticeTraits<D>::Q> weights;
    std::vector<std::pair<size_t, BoundaryType>> boundaries;
    std::vector<Wall<T,D>> walls;
    std::unordered_map<size_t, std::array<T,D>> inletVelocities;
    std::unordered_map<size_t, T> outletPressures;
    std::vector<std::array<T, LatticeTraits<D>::Q>> previousDistributions;
    std::vector<IBMNode<T,D>> ibmNodes;

    void applyIBMForces() {
        auto& nodes = mesh.getNodes();
        
        #pragma omp parallel for schedule(dynamic, 128)
        for(size_t i = 0; i < ibmNodes.size(); i++) {
            auto& ibmNode = ibmNodes[i];
            std::array<T,D> interpolatedVelocity = {};
            T totalWeight = 0;
            
            for(size_t j = 0; j < ibmNode.nearNodes.size(); j++) {
                size_t nodeIdx = ibmNode.nearNodes[j];
                if(nodeIdx >= nodes.size() || !nodes[nodeIdx].isActive) continue;
                
                T rho = computeDensity(nodes[nodeIdx].distributions);
                auto u = computeVelocity(nodes[nodeIdx].distributions, rho);
                T weight = ibmNode.weights[j];
                
                for(size_t d = 0; d < D; d++) {
                    interpolatedVelocity[d] += weight * u[d];
                }
                totalWeight += weight;
            }
            
            if(totalWeight > 0) {
                for(size_t d = 0; d < D; d++) {
                    interpolatedVelocity[d] /= totalWeight;
                    ibmNode.force[d] = (ibmNode.velocity[d] - interpolatedVelocity[d]) / deltaT;
                }
            }
        }
        
        #pragma omp parallel for schedule(dynamic, 128)
        for(size_t i = 0; i < ibmNodes.size(); i++) {
            const auto& ibmNode = ibmNodes[i];
            for(size_t j = 0; j < ibmNode.nearNodes.size(); j++) {
                size_t nodeIdx = ibmNode.nearNodes[j];
                if(nodeIdx >= nodes.size() || !nodes[nodeIdx].isActive) continue;
                
                T weight = ibmNode.weights[j];
                for(size_t k = 0; k < LatticeTraits<D>::Q; k++) {
                    T force = 0;
                    for(size_t d = 0; d < D; d++) {
                        force += ibmNode.force[d] * LatticeTraits<D>::directions[k][d];
                    }
                    nodes[nodeIdx].distributions[k] += weight * force * weights[k];
                }
            }
        }
    }

public:
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

    T computeStrainRateMagnitude(const std::array<T,D>& u, 
                                const std::array<T, LatticeTraits<D>::Q>& f,
                                const std::array<T, LatticeTraits<D>::Q>& feq) const {
        std::array<std::array<T,D>, D> S = {};
        for(size_t a = 0; a < D; ++a) {
            for(size_t b = 0; b < D; ++b) {
                for(size_t i = 0; i < LatticeTraits<D>::Q; ++i) {
                    S[a][b] += LatticeTraits<D>::directions[i][a] * 
                              LatticeTraits<D>::directions[i][b] * 
                              (f[i] - feq[i]);
                }
                S[a][b] *= -1.0/(2.0 * cs2);
            }
        }

        T magnitude = 0;
        for(size_t a = 0; a < D; ++a) {
            for(size_t b = 0; b < D; ++b) {
                magnitude += S[a][b] * S[a][b];
            }
        }
        return std::sqrt(2.0 * magnitude);
    }

    void stabilizeDistributions(size_t nodeIdx) {
        auto& node = mesh.getNodes()[nodeIdx];
        
        for(auto& f : node.distributions) {
            f = std::max(f, T(1e-10));
        }

        T rho = computeDensity(node.distributions);
        if(rho < minRho || rho > maxRho) {
            T scale = T(1.0) / rho;
            for(auto& f : node.distributions) {
                f *= scale;
            }
        }
    }

    T computeWallDistance(const std::array<T,D>& point, const Wall<T,D>& wall) const {
        T distance = 0;
        for(size_t d = 0; d < D; d++) {
            T diff = point[d] - wall.position[d];
            distance += diff * wall.normal[d];
        }
        return std::abs(distance);
    }

    void collision() {
        auto& nodes = mesh.getNodes();
        #pragma omp parallel for schedule(dynamic, 128)
        for(size_t i = 0; i < nodes.size(); i++) {
            if(!nodes[i].isActive) continue;
            
            T rho = computeDensity(nodes[i].distributions);
            auto u = computeVelocity(nodes[i].distributions, rho);
            
            T magU = 0;
            for(size_t d = 0; d < D; d++) {
                magU += u[d] * u[d];
            }
            magU = std::sqrt(magU);
            if(magU > maxVelocity) {
                T scale = maxVelocity / magU;
                for(size_t d = 0; d < D; d++) {
                    u[d] *= scale;
                }
            }
            
            auto feq = computeEquilibrium(rho, u);
            
            T strainRate = computeStrainRateMagnitude(u, nodes[i].distributions, feq);
            T nuT = (Cs * deltaX) * (Cs * deltaX) * strainRate;
            
            T tau_eff = 0.5 + (nuT + cs2 * (1.0/omega - 0.5)) / cs2;
            T omega_eff = 1.0 / tau_eff;
            
            const T alpha = 0.7;
            for(size_t k = 0; k < LatticeTraits<D>::Q; k++) {
                T df = feq[k] - nodes[i].distributions[k];
                nodes[i].distributions[k] += alpha * omega_eff * df;
            }
            
            stabilizeDistributions(i);
        }
    }

    void streaming() {
        auto& nodes = mesh.getNodes();
        previousDistributions.resize(nodes.size());
        
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < nodes.size(); i++) {
            previousDistributions[i] = nodes[i].distributions;
        }
        
        #pragma omp parallel for schedule(dynamic, 128)
        for(size_t i = 0; i < nodes.size(); i++) {
            if(!nodes[i].isActive) continue;
            
            for(size_t k = 0; k < LatticeTraits<D>::Q; k++) {
                size_t neighborIdx = mesh.getNeighborIndex(i, k);
                if(neighborIdx < nodes.size()) {
                    nodes[neighborIdx].distributions[k] = previousDistributions[i][k];
                }
            }
        }
    }

    std::array<T, LatticeTraits<D>::Q> computeEquilibrium(T rho, const std::array<T,D>& u) const {
        std::array<T, LatticeTraits<D>::Q> feq;
        T usqr = 0;
        for(size_t d = 0; d < D; d++) {
            usqr += u[d] * u[d];
        }
        
        for(size_t i = 0; i < LatticeTraits<D>::Q; i++) {
            T cu = 0;
            for(size_t d = 0; d < D; d++) {
                cu += LatticeTraits<D>::directions[i][d] * u[d];
            }
            feq[i] = weights[i] * rho * (1 + 3*cu + 4.5*cu*cu - 1.5*usqr);
            feq[i] = std::max(feq[i], T(1e-10));
        }
        return feq;
    }

    void applyBoundaryConditions() {
        auto& nodes = mesh.getNodes();
        
        #pragma omp parallel for schedule(dynamic, 64)
        for(size_t i = 0; i < boundaries.size(); i++) {
            const auto& [nodeIdx, type] = boundaries[i];
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
            stabilizeDistributions(nodeIdx);
        }

        #pragma omp parallel for schedule(dynamic, 128)
        for(size_t i = 0; i < nodes.size(); i++) {
            if(!nodes[i].isActive) continue;
            
            for(const auto& wall : walls) {
                T distance = computeWallDistance(nodes[i].position, wall);
                if(distance < 1.5 * deltaX) {
                    applyBounceBack(i);
                    break;
                }
            }
        }
    }

    void applyBounceBack(size_t idx) {
        auto& node = mesh.getNodes()[idx];
        auto temp = node.distributions;
        for(size_t i = 1; i < LatticeTraits<D>::Q; i++) {
            node.distributions[i] = temp[LatticeTraits<D>::Q - i];
        }
    }

    void applyZouHeVelocity(size_t idx, const std::array<T,D>& velocity) {
        auto& node = mesh.getNodes()[idx];
        T rho = computeDensity(node.distributions);
        auto feq = computeEquilibrium(rho, velocity);
        
        const T beta = 0.8;
        for(size_t i = 0; i < LatticeTraits<D>::Q; i++) {
            node.distributions[i] = (1 - beta) * node.distributions[i] + beta * feq[i];
        }
    }

    void applyZouHePressure(size_t idx, T pressure) {
        auto& node = mesh.getNodes()[idx];
        auto velocity = computeVelocity(node.distributions, pressure);
        auto feq = computeEquilibrium(pressure, velocity);
        
        const T beta = 0.8;
        for(size_t i = 0; i < LatticeTraits<D>::Q; i++) {
            node.distributions[i] = (1 - beta) * node.distributions[i] + beta * feq[i];
        }
    }

public:
    LBMSolver(MeshObj<T,D>& mesh_, T viscosity, T dx = 1.0, T dt = 1.0) 
        : mesh(mesh_), deltaX(dx), deltaT(dt) {
        cs2 = 1.0/3.0;
        T tau = std::max(viscosity/(cs2 * deltaT) + 0.5, 0.51);
        omega = 1.0/tau;
        Cs = 0.17;
        minRho = 0.1;
        maxRho = 2.0;
        maxVelocity = 0.1 * std::sqrt(cs2);
        initializeWeights();
    }

    void initialize(T rho0 = 1.0, const std::array<T,D>& u0 = {}) {
        auto& nodes = mesh.getNodes();
        for(auto& node : nodes) {
            if(!node.isActive) continue;
            node.distributions = computeEquilibrium(rho0, u0);
        }
        previousDistributions.resize(nodes.size());
    }

    void addIBMNode(const IBMNode<T,D>& node) {
        ibmNodes.push_back(node);
    }

    void addIBMNode(const std::array<T,D>& position, const std::array<T,D>& velocity) {
        IBMNode<T,D> node;
        node.position = position;
        node.velocity = velocity;
        node.force.fill(0);
        ibmNodes.push_back(node);
    }

    void collideAndStream() {
        applyBoundaryConditions();
        applyIBMForces();
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
        for(size_t d = 0; d < D; d++) {
            velocity[d] /= std::max(rho, minRho);
        }
        return velocity;
    }

    void addWall(const Wall<T,D>& wall) {
        walls.push_back(wall);
    }

    void addWallPlane(const std::array<T,D>& point, const std::array<T,D>& normal) {
        Wall<T,D> wall;
        wall.position = point;
        
        T norm = 0;
        for(size_t d = 0; d < D; d++) {
            norm += normal[d] * normal[d];
        }
        norm = std::sqrt(norm);
        
        for(size_t d = 0; d < D; d++) {
            wall.normal[d] = normal[d] / norm;
        }
        
        walls.push_back(wall);
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

    void setSmagorinskyConstant(T Cs_) { Cs = Cs_; }
    void setDensityLimits(T min, T max) { minRho = min; maxRho = max; }
    void setMaxVelocity(T max) { maxVelocity = max; }
};

#endif // LBM_SOLVER_HPP
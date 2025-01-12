#include "../application/LatticeBoltz/LBMSolver.hpp"
#include "../application/PostProcess/Visual.hpp"
#include <cmath>
#include <fstream>

template<typename T>
class CylinderFlow {
private:
    static constexpr size_t D = 2;  // 2D simulation
    static constexpr size_t Nx = 400;
    static constexpr size_t Ny = 100;
    static constexpr T Re = 100;     // Lower Reynolds number for stability
    static constexpr T U0 = 0.1;     // Lower inlet velocity for stability
    static constexpr T R = 10.0;     // Cylinder radius
    
    std::array<size_t, D> dims;
    std::unique_ptr<MeshObj<T,D>> mesh;
    std::unique_ptr<LBMSolver<T,D>> solver;
    std::ofstream forceLog;
    
public:
    CylinderFlow() : dims({Nx, Ny}) {
        T dx = 1.0;
        T viscosity = U0 * (2*R) / Re;
        
        // Initialize mesh and solver
        mesh = std::make_unique<MeshObj<T,D>>(dims, dx);
        solver = std::make_unique<LBMSolver<T,D>>(*mesh, viscosity);
        
        forceLog.open("cylinder_forces.txt");
        setupBoundaries();
        addCylinder();
    }
    
    ~CylinderFlow() {
        if(forceLog.is_open()) forceLog.close();
    }
    
    void setupBoundaries() {
        // Inlet (left boundary)
        for(size_t j = 0; j < Ny; j++) {
            size_t idx = j * Nx;
            solver->setBoundary(idx, BoundaryType::VelocityInlet);
            solver->setInletVelocity(idx, {U0, 0.0});
        }
        
        // Outlet (right boundary)
        for(size_t j = 0; j < Ny; j++) {
            size_t idx = j * Nx + (Nx-1);
            solver->setBoundary(idx, BoundaryType::PressureOutlet);
            solver->setOutletPressure(idx, 1.0);
        }
        
        // Top and bottom walls
        for(size_t i = 0; i < Nx; i++) {
            solver->setBoundary(i, BoundaryType::NoSlip);
            solver->setBoundary(i + (Ny-1)*Nx, BoundaryType::NoSlip);
        }
    }
    
private:
    void addCylinder() {
        // Center of cylinder
        T cx = Nx/4.0;
        T cy = Ny/2.0;
        
        // Add IBM nodes around cylinder surface
        size_t nPoints = 100;
        for(size_t i = 0; i < nPoints; i++) {
            T theta = 2.0 * M_PI * i / nPoints;
            
            IBMNode<T,D> node;
            node.position = {
                cx + R * std::cos(theta),
                cy + R * std::sin(theta)
            };
            node.velocity = {0.0, 0.0};  // Stationary cylinder
            node.force = {0.0, 0.0};
            
            // Find nearby fluid nodes
            const auto& meshNodes = mesh->getNodes();
            for(size_t j = 0; j < meshNodes.size(); j++) {
                std::array<T,D> r;
                for(size_t d = 0; d < D; d++) {
                    r[d] = meshNodes[j].position[d] - node.position[d];
                }
                T weight = solver->deltaFunction(r);
                if(weight > 0) {
                    node.nearNodes.push_back(j);
                    node.weights.push_back(weight);
                }
            }
            
            solver->addIBMNode(node);
        }
    }

public:
    void run(size_t nSteps, size_t saveInterval) {
        // Initialize flow field
        solver->initialize(1.0, {U0, 0.0});
        
        // Time stepping
        for(size_t step = 0; step < nSteps; step++) {
            solver->collideAndStream();
            
            if(step % saveInterval == 0) {
                std::string filename = "cylinder_" + std::to_string(step);
                Visual<T,D>::writeVTK(*mesh, filename);
                
                // Log forces
                const auto& ibmNodes = solver->getIBMNodes();
                T fx = 0.0, fy = 0.0;
                for(const auto& node : ibmNodes) {
                    fx += node.force[0];
                    fy += node.force[1];
                }
                forceLog << step << " " << fx << " " << fy << "\n";
            }
        }
    }
};

int main() {
    CylinderFlow<double> simulation;
    simulation.run(100000, 1000);  // Run for 10000 steps, save every 100 steps
    return 0;
}
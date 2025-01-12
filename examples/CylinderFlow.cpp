#include "../application/LatticeBoltz/LBMSolver.hpp"
#include "../application/PostProcess/Visual.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>

template<typename T>
class CylinderFlow {
private:
    static constexpr size_t D = 2;
    static constexpr size_t Nx = 400;
    static constexpr size_t Ny = 100;
    static constexpr T Re = 600;     // Reynolds number
    static constexpr T U0 = 40.0;     // Inlet velocity
    static constexpr T R = 10.0;     // Cylinder radius
    static constexpr T CX = Nx/4.0;  // Cylinder center x
    static constexpr T CY = Ny/2.0;  // Cylinder center y
    
    std::array<size_t, D> dims;
    std::unique_ptr<MeshObj<T,D>> mesh;
    std::unique_ptr<LBMSolver<T,D>> solver;
    std::ofstream forceLog;
    
public:
    CylinderFlow() : dims({Nx, Ny}) {
        T dx = 1;
        T deltaT = 0.8;
        T viscosity = U0 * (2*R) / Re;
        
        mesh = std::make_unique<MeshObj<T,D>>(dims, dx);
        solver = std::make_unique<LBMSolver<T,D>>(*mesh, viscosity, dx, deltaT);
        
        forceLog.open("cylinder_forces.txt");
        forceLog << std::scientific << std::setprecision(6);
        
        setupBoundaries();
        addCylinder();
    }
    
    ~CylinderFlow() {
        if(forceLog.is_open()) forceLog.close();
    }

private:
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
    
    void addCylinder() {
        size_t nPoints = 100;
        for(size_t i = 0; i < nPoints; i++) {
            T theta = 2.0 * M_PI * i / nPoints;
            
            IBMNode<T,D> node;
            node.position = {
                CX + R * std::cos(theta),
                CY + R * std::sin(theta)
            };
            node.velocity = {0.0, 0.0};
            node.force = {0.0, 0.0};
            
            // Find nearby fluid nodes
            const auto& meshNodes = mesh->getNodes();
            for(size_t j = 0; j < meshNodes.size(); j++) {
                T dx = meshNodes[j].position[0] - node.position[0];
                T dy = meshNodes[j].position[1] - node.position[1];
                T dist = std::sqrt(dx*dx + dy*dy);
                
                if(dist <= 2.0) {
                    node.nearNodes.push_back(j);
                    node.weights.push_back((1 + std::cos(M_PI*dist/2))/2);
                }
            }
            
            solver->addIBMNode(node);
        }
    }
    
public:
    void run(size_t nSteps, size_t saveInterval) {
        solver->initialize(1.0, {U0, 0.0});
        
        for(size_t step = 0; step < nSteps; step++) {
            solver->collideAndStream();
            
            if(step % saveInterval == 0) {
                // Save flow field
                std::string filename = "cylinder_" + std::to_string(step/saveInterval);
                Visual<T,D>::writeVTK(*mesh, filename);
                
                // Log forces
                const auto& ibmNodes = solver->getIBMNodes();
                T fx = 0.0, fy = 0.0;
                for(const auto& node : ibmNodes) {
                    fx += node.force[0];
                    fy += node.force[1];
                }
                
                // Compute coefficients
                T rho = 1.0;
                T cd = 2*fx/(rho*U0*U0*2*R);
                T cl = 2*fy/(rho*U0*U0*2*R);
                
                forceLog << step << " " << cd << " " << cl << "\n";
            }
        }
    }
};

int main() {
    CylinderFlow<double> simulation;
    simulation.run(10000, 10);  // Run for 100k steps, save every 1000 steps
    return 0;
}
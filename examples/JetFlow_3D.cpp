#include "../application/LatticeBoltz/LBMSolver.hpp"
#include "../application/PostProcess/Visual.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>

template<typename T>
class JetFlow {
private:
    static constexpr size_t D = 3;           // Change to 3D
    static constexpr size_t Nx = 100;        // Length
    static constexpr size_t Ny = 100;        // Height
    static constexpr size_t Nz = 100;        // Width
    static constexpr T Re = 60;              // Reynolds number
    static constexpr T U0 = 1.2;             // Jet velocity
    static constexpr T JetRadius = 10.0;     // Jet inlet radius
    
    std::array<size_t, D> dims;
    std::unique_ptr<MeshObj<T,D>> mesh;
    std::unique_ptr<LBMSolver<T,D>> solver;
    std::ofstream dataLog;
    
public:
    JetFlow() : dims({Nx, Ny, Nz}) {
        T dx = 0.01;
        T deltaT = 0.01;
        T viscosity = U0 * (2*JetRadius) / Re;
        
        mesh = std::make_unique<MeshObj<T,D>>(dims, dx);
        solver = std::make_unique<LBMSolver<T,D>>(*mesh, viscosity, dx, deltaT);
        
        dataLog.open("jet3d_data.txt");
        dataLog << std::scientific << std::setprecision(6);
        
        setupBoundaries();
    }
    
    ~JetFlow() {
        if(dataLog.is_open()) dataLog.close();
    }

private:
    void setupBoundaries() {
        // Inlet (center of YZ plane)
        T centerY = Ny/2.0;
        T centerZ = Nz/2.0;
        
        for(size_t j = 0; j < Ny; j++) {
            for(size_t k = 0; k < Nz; k++) {
                size_t idx = k * Nx * Ny + j * Nx;
                T dy = j - centerY;
                T dz = k - centerZ;
                T dist = std::sqrt(dy*dy + dz*dz);
                
                if(dist <= JetRadius) {
                    solver->setBoundary(idx, BoundaryType::VelocityInlet);
                    solver->setInletVelocity(idx, {U0, 0.0, 0.0});
                } else {
                    solver->setBoundary(idx, BoundaryType::NoSlip);
                }
            }
        }
        
        // Outlet (right boundary)
        for(size_t j = 0; j < Ny; j++) {
            for(size_t k = 0; k < Nz; k++) {
                size_t idx = k * Nx * Ny + j * Nx + (Nx-1);
                solver->setBoundary(idx, BoundaryType::PressureOutlet);
                solver->setOutletPressure(idx, 1.0);
            }
        }
        
        // All other walls
        // Top and bottom walls
        for(size_t i = 0; i < Nx; i++) {
            for(size_t k = 0; k < Nz; k++) {
                solver->setBoundary(k * Nx * Ny + i, BoundaryType::NoSlip);
                solver->setBoundary(k * Nx * Ny + (Ny-1) * Nx + i, BoundaryType::NoSlip);
            }
        }
        
        // Front and back walls
        for(size_t i = 0; i < Nx; i++) {
            for(size_t j = 0; j < Ny; j++) {
                solver->setBoundary(j * Nx + i, BoundaryType::NoSlip);
                solver->setBoundary((Nz-1) * Nx * Ny + j * Nx + i, BoundaryType::NoSlip);
            }
        }
    }
    
    void logData(size_t step) {
        const auto& nodes = mesh->getNodes();
        T maxVel = 0.0;
        T centerlineVel = 0.0;
        
        // Sample along centerline
        size_t j = Ny/2;
        size_t k = Nz/2;
        for(size_t i = 0; i < Nx; i++) {
            size_t idx = k * Nx * Ny + j * Nx + i;
            if(!nodes[idx].isActive) continue;
            
            T rho = solver->computeDensity(nodes[idx].distributions);
            auto vel = solver->computeVelocity(nodes[idx].distributions, rho);
            T speed = std::sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
            
            maxVel = std::max(maxVel, speed);
            if(i == Nx/2) centerlineVel = speed;
        }
        
        dataLog << step << " " << maxVel << " " << centerlineVel << "\n";
    }
    
public:
    void run(size_t nSteps, size_t saveInterval) {
        solver->initialize(1.0);
        
        for(size_t step = 0; step < nSteps; step++) {
            solver->collideAndStream();
            
            if(step % saveInterval == 0) {
                std::string filename = "jet3d_" + std::to_string(step/saveInterval);
                Visual<T,D>::writeVTK(*mesh, filename);
                logData(step);
            }
        }
    }
};

int main() {
    JetFlow<double> simulation;
    simulation.run(1000, 100);  // Run for 10k steps, save every 100 steps
    return 0;
}
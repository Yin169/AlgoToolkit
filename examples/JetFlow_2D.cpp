#include "../application/LatticeBoltz/LBMSolver.hpp"
#include "../application/PostProcess/Visual.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>

template<typename T>
class JetFlow {
private:
    static constexpr size_t D = 2;
    static constexpr size_t Nx = 400;  // Length
    static constexpr size_t Ny = 400;  // Height
    static constexpr T Re = 60;       // Reynolds number
    static constexpr T U0 = 1.0;       // Jet velocity
    static constexpr T JetWidth = 20;  // Jet inlet width
    
    std::array<size_t, D> dims;
    std::unique_ptr<MeshObj<T,D>> mesh;
    std::unique_ptr<LBMSolver<T,D>> solver;
    std::ofstream dataLog;
    
public:
    JetFlow() : dims({Nx, Ny}) {
        T dx = 0.01;
        T deltaT = 0.01;
        T viscosity = U0 * JetWidth / Re;
        
        mesh = std::make_unique<MeshObj<T,D>>(dims, dx);
        solver = std::make_unique<LBMSolver<T,D>>(*mesh, viscosity, dx, deltaT);
        
        dataLog.open("jet_data.txt");
        dataLog << std::scientific << std::setprecision(6);
        
        setupBoundaries();
    }
    
    ~JetFlow() {
        if(dataLog.is_open()) dataLog.close();
    }

private:
    void setupBoundaries() {
        // Inlet (center of left boundary)
        size_t jetStart = (Ny - JetWidth)/2;
        size_t jetEnd = jetStart + JetWidth;
        
        for(size_t j = 0; j < Ny; j++) {
            size_t idx = j * Nx;
            if(j >= jetStart && j < jetEnd) {
                solver->setBoundary(idx, BoundaryType::VelocityInlet);
                solver->setInletVelocity(idx, {U0, 0.0});
            } else {
                solver->setBoundary(idx, BoundaryType::NoSlip);
            }
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
            solver->setBoundary((Ny-1)*Nx + i, BoundaryType::NoSlip);
        }
    }
    
    void logData(size_t step) {
        const auto& nodes = mesh->getNodes();
        T maxVel = 0.0;
        T centerlineVel = 0.0;
        
        // Sample along centerline
        size_t j = Ny/2;
        for(size_t i = 0; i < Nx; i++) {
            size_t idx = j * Nx + i;
            if(!nodes[idx].isActive) continue;
            
            T rho = solver->computeDensity(nodes[idx].distributions);
            auto vel = solver->computeVelocity(nodes[idx].distributions, rho);
            T speed = std::sqrt(vel[0]*vel[0] + vel[1]*vel[1]);
            
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
                std::string filename = "jet_" + std::to_string(step/saveInterval);
                Visual<T,D>::writeVTK(*mesh, filename);
                logData(step);
            }
        }
    }
};

int main() {
    JetFlow<double> simulation;
    simulation.run(1000, 6);  // Run for 50k steps, save every 500 steps
    return 0;
}
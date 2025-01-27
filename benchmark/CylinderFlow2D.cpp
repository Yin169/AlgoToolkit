#include "LBMSolver.hpp"
#include "Visual.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <chrono>

// Simulation parameters
constexpr double scale = 1.0;
constexpr double Re = 3900;            // Reynolds number
constexpr double Ma = 0.1;            // Mach number
const double U0 = Ma * std::sqrt(1.0/3.0); // Inlet velocity (Ma * cs)
constexpr double D = 70 * scale;              // Cylinder diameter in lattice units
const double nu = U0 * D / Re;    // Kinematic viscosity
constexpr size_t Nx = 800 * scale;            // Domain size in x
constexpr size_t Ny = 400 * scale;            // Domain size in y
constexpr double cx = Nx/6;           // Cylinder center x
constexpr double cy = Ny/2;           // Cylinder center y

// Function to check if a point is inside the cylinder
// Fix the isInsideCylinder function
bool isInsideCylinder(double x, double y) {
    double dx = x - cx;
    double dy = y - cy;
    return (dx*dx + dy*dy) <= (D*D/4.0);  // Correct circular check
}

int main() {
    std::cout << "Initializing cylinder flow simulation...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create mesh
    std::array<size_t, 2> dims = {Nx, Ny};
    MeshObj<double, 2> mesh(dims, 1.0);
    
    // Initialize solver
    LBMSolver<double, 2> solver(mesh, nu);
    
    // Initialize flow field
    solver.initialize(1.0 , {U0, 0.0});
    
    // Set boundary conditions
    std::cout << "Setting up boundary conditions...\n";
    
    // Top and bottom walls
    for(size_t i = 0; i < Nx; i++) {
        solver.setBoundary(i, BoundaryType::NoSlip);                  // Bottom wall
        solver.setBoundary(i + (Ny-1)*Nx, BoundaryType::NoSlip);     // Top wall
    }
    
    // Inlet (left boundary)
    for(size_t j = 0; j < Ny; j++) {
        size_t idx = j * Nx;
        solver.setBoundary(idx, BoundaryType::VelocityInlet);
        solver.setInletVelocity(idx, {U0, 0.0});
    }
    
    // Outlet (right boundary)
    for(size_t j = 0; j < Ny; j++) {
        size_t idx = (Nx-1) + j * Nx;
        solver.setBoundary(idx, BoundaryType::PressureOutlet);
        solver.setOutletPressure(idx, 1.0);
    }
    
    // Replace IBM nodes with wall boundary for cylinder
    std::cout << "Creating cylinder boundary...\n";
    // Remove or replace this line as it's incorrect for a cylinder
    // solver.addWallPlane({cx, cy}, {1.0, 0.0});
    
    // Set nodes inside cylinder as inactive
    auto& nodes = mesh.getNodes();
    for(size_t j = 0; j < Ny; j++) {
        for(size_t i = 0; i < Nx; i++) {
            if(isInsideCylinder(i, j)) {
                nodes[i + j*Nx].isActive = false;
            }
        }
    }

    // Add stability parameters
    solver.setDensityLimits(0.1, 2.0);
    solver.setMaxVelocity(0.15);
    solver.setSmagorinskyConstant(0.17);
    
    // Simulation parameters
    const int nSteps = 1000000;
    const int saveInterval = 100;
    
    // Initialize visualization
    Visual<double, 2> visualizer;
    
    std::cout << "Starting simulation...\n";
    std::cout << "Reynolds number: " << Re << "\n";
    std::cout << "Mach number: " << Ma << "\n";
    std::cout << "Inlet velocity: " << U0 << "\n";
    std::cout << "Kinematic viscosity: " << nu << "\n";
    
    // Main simulation loop
    for(int step = 0; step < nSteps; step++) {
        solver.collideAndStream();
        
        if(step % saveInterval == 0) {
            std::cout << "Step " << step << "/" << nSteps;
            
            // Calculate and print maximum velocity
            double maxVel = 0.0;
            for(size_t i = 0; i < nodes.size(); i++) {
                if(!nodes[i].isActive) continue;
                auto rho = solver.computeDensity(nodes[i].distributions);
                auto vel = solver.computeVelocity(nodes[i].distributions, rho);
                double velMag = std::sqrt(vel[0]*vel[0] + vel[1]*vel[1]);
                maxVel = std::max(maxVel, velMag);
            }
            std::cout << ", Max velocity: " << maxVel << "\n";
            
            // Save visualization
            std::string filename = "cylinder_flow_" + std::to_string(step/saveInterval);
            Visual<double, 2>::writeVTK(mesh, filename);
        }
    }
    
    // Calculate and print simulation time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "\nSimulation completed in " << duration.count() << " seconds\n";
    
    return 0;
}
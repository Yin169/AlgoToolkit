#include "../application/LatticeBoltz/LBMSolver.hpp"
#include "../application/PostProcess/Visual.hpp"
#include <iostream>
#include <fstream>
#include <string>

int main() {
    // Domain parameters
    const size_t nx = 400;  // Domain width
    const size_t ny = 200;  // Domain height
    const double dx = 1.0;  // Grid spacing
    const double dt = 1.0;  // Time step
    const double viscosity = 0.01;  // Fluid viscosity
    const double jetVelocity = 2.0;  // Inlet jet velocity
    const size_t maxSteps = 10000;   // Maximum simulation steps
    const size_t saveInterval = 100;  // Save results every N steps

    // Create mesh
    MeshObj<double, 2> mesh({nx, ny}, 1.0);
    
    // Initialize LBM solver
    LBMSolver<double, 2> solver(mesh, viscosity, dx, dt);
    
    // Enable Smagorinsky turbulence model
    solver.setSmagorinskyConstant(0.17);  // Typical value for Cs

    // Set initial conditions (fluid at rest)
    solver.initialize(1.0, {0.0, 0.0});
    
    // Set up jet inlet (middle portion of left boundary)
    const size_t jetStart = ny/3;
    const size_t jetEnd = 2*ny/3;
    for(size_t j = 0; j < ny; j++) {
        size_t idx = j * nx;
        if (j >= jetStart && j <= jetEnd) {
            solver.setBoundary(idx, BoundaryType::VelocityInlet);
            solver.setInletVelocity(idx, {jetVelocity, 0.0});
        } else {
            solver.setBoundary(idx, BoundaryType::NoSlip);
        }
    }
    
    // Set up outlet (right boundary)
    for(size_t j = 0; j < ny; j++) {
        size_t idx = j * nx + (nx-1);
        solver.setBoundary(idx, BoundaryType::PressureOutlet);
        solver.setOutletPressure(idx, 1.0);
    }
    
    // Set up walls (top and bottom boundaries)
    for(size_t i = 0; i < nx; i++) {
        // Bottom wall
        solver.setBoundary(i, BoundaryType::PressureOutlet);
        solver.setOutletPressure(i, 1.0);
        // Top wall
        solver.setBoundary((ny-1) * nx + i, BoundaryType::PressureOutlet);
        solver.setOutletPressure((ny-1) * nx + i, 1.0);
    }

    // Simulation loop
    for(size_t step = 0; step < maxSteps; step++) {
        solver.collideAndStream();
        
        // Save results periodically in VTK format
        if(step % saveInterval == 0) {
            std::string filename = "jetflow_" + std::to_string(step);
            Visual<double, 2>::writeVTK(mesh, filename);
            
            std::cout << "Step " << step << " completed\n";
        }
    }
    
    return 0;
}
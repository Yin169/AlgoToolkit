#include "LBMSolver.hpp"
#include "Visual.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <chrono>

// Simulation parameters
constexpr double Re = 60;            // Reynolds number
constexpr double Ma = 0.1;            // Mach number
const double U0 = Ma * std::sqrt(1.0/3.0); // Inlet velocity (Ma * cs)
constexpr double D = 40;              // Cylinder diameter in lattice units
const double nu = U0 * D / Re;    // Kinematic viscosity
constexpr size_t Nx = 1600;            // Domain size in x
constexpr size_t Ny = 800;            // Domain size in y
constexpr double cx = Nx/4;           // Cylinder center x
constexpr double cy = Ny/2;           // Cylinder center y

// Function to check if a point is inside the cylinder
bool isInsideCylinder(double x, double y) {
    double dx = x - cx;
    double dy = y - cy;
    return (dx*dx + dy*dy) <= (D*D/4.0);
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
    solver.initialize();
    
    // Set boundary conditions
    std::cout << "Setting up boundary conditions...\n";
    
    // Top and bottom walls
    for(size_t i = 0; i < Nx; i++) {
        // solver.setBoundary(i, BoundaryType::NoSlip);                  // Bottom wall
        // solver.setBoundary(i + (Ny-1)*Nx, BoundaryType::NoSlip);     // Top wall

        solver.setBoundary(i, BoundaryType::PressureOutlet);
        solver.setOutletPressure(i, 1.0);
        solver.setBoundary(i + (Ny-1)*Nx, BoundaryType::PressureOutlet);
        solver.setOutletPressure(i + (Ny-1)*Nx, 1.0);
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
        // solver.setBoundary(idx, BoundaryType::VelocityInlet);
        // solver.setInletVelocity(idx, {0.0, 0.0});
        solver.setBoundary(idx, BoundaryType::PressureOutlet);
        solver.setOutletPressure(idx, 1.0);
    }
    
    // Create cylinder using IBM nodes
    std::cout << "Creating cylinder boundary...\n";
    constexpr int numIBMNodes = 120;  // Number of points around cylinder
    for(int i = 0; i < numIBMNodes; i++) {
        double theta = 2.0 * M_PI * i / numIBMNodes;
        IBMNode<double, 2> node;
        node.position = {cx + D/2.0 * std::cos(theta), 
                        cy + D/2.0 * std::sin(theta)};
        node.velocity = {0.0, 0.0};  // Stationary cylinder
        node.force = {0.0, 0.0};
        
        // Find near nodes and compute weights
        for(int dx = -2; dx <= 2; dx++) {
            for(int dy = -2; dy <= 2; dy++) {
                int x = static_cast<int>(node.position[0]) + dx;
                int y = static_cast<int>(node.position[1]) + dy;
                if(x >= 0 && x < Nx && y >= 0 && y < Ny) {
                    size_t idx = x + y * Nx;
                    double dist = std::sqrt(std::pow(x - node.position[0], 2) + 
                                          std::pow(y - node.position[1], 2));
                    if(dist <= 2.0) {
                        node.nearNodes.push_back(idx);
                        node.weights.push_back(std::exp(-dist*dist/2.0));
                    }
                }
            }
        }
        solver.addIBMNode(node);
    }
    
    // Set nodes inside cylinder as inactive
    auto& nodes = mesh.getNodes();
    for(size_t j = 0; j < Ny; j++) {
        for(size_t i = 0; i < Nx; i++) {
            if(isInsideCylinder(i, j)) {
                nodes[i + j*Nx].isActive = false;
            }
        }
    }
    
    // Simulation parameters
    const int nSteps = 100000;
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
#include "LBMSolver.hpp"
#include "Visual.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <chrono>

// Simulation parameters
constexpr double scale = 1.0;
constexpr double Re = 2000;            // Reynolds number
constexpr double Ma = 0.002;            // Mach number
const double U0 = Ma * std::sqrt(1.0/3.0); // Inlet velocity (Ma * cs)
constexpr double D = 40 * scale;              // Cylinder diameter in lattice units
const double nu = U0 * D / Re;    // Kinematic viscosity
constexpr size_t Nx = 400 * scale;            // Domain size in x
constexpr size_t Ny = 200 * scale;            // Domain size in y
constexpr size_t Nz = 2 * scale;            // Domain size in z
constexpr double cx = Nx/4;           // Cylinder center x
constexpr double cy = Ny/2;           // Cylinder center y
constexpr double cz = Nz/2;           // Cylinder center z

// Function to check if a point is inside the cylinder
bool isInsideCylinder(double x, double y, double z) {
    double dx = x - cx;
    double dy = y - cy;
    return (dx*dx + dy*dy) <= (D*D/4.0); // Cylinder extends in z-direction
}

int main() {
    std::cout << "Initializing 3D cylinder flow simulation...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create 3D mesh
    std::array<size_t, 3> dims = {Nx, Ny, Nz};
    MeshObj<double, 3> mesh(dims, 1.0);
    
    // Initialize 3D solver
    LBMSolver<double, 3> solver(mesh, nu);
    
    // Initialize flow field
    solver.initialize();
    
    std::cout << "Setting up boundary conditions...\n";
    
    // Set boundary conditions for all walls
    // Front and back walls (z-direction)
    for(size_t i = 0; i < Nx; i++) {
        for(size_t j = 0; j < Ny; j++) {
            size_t front_idx = i + j * Nx;
            size_t back_idx = i + j * Nx + (Nz-1) * Nx * Ny;
            solver.setBoundary(front_idx, BoundaryType::PressureOutlet);
            solver.setOutletPressure(front_idx, 1.0);
            solver.setBoundary(back_idx, BoundaryType::PressureOutlet);
            solver.setOutletPressure(back_idx, 1.0);
        }
    }
    
    // Top and bottom walls (y-direction)
    for(size_t i = 0; i < Nx; i++) {
        for(size_t k = 0; k < Nz; k++) {
            size_t bottom_idx = i + k * Nx * Ny;
            size_t top_idx = i + (Ny-1) * Nx + k * Nx * Ny;
            solver.setBoundary(bottom_idx, BoundaryType::PressureOutlet);
            solver.setOutletPressure(bottom_idx, 1.0);
            solver.setBoundary(top_idx, BoundaryType::PressureOutlet);
            solver.setOutletPressure(top_idx, 1.0);
        }
    }
    
    // Inlet (left boundary)
    for(size_t j = 0; j < Ny; j++) {
        for(size_t k = 0; k < Nz; k++) {
            size_t idx = j * Nx + k * Nx * Ny;
            solver.setBoundary(idx, BoundaryType::PressureOutlet);
            solver.setOutletPressure(idx, 1.1);
            // solver.setBoundary(idx, BoundaryType::VelocityInlet);
            // solver.setInletVelocity(idx, {U0, 0.0, 0.0});
        }
    }
    
    // Outlet (right boundary)
    for(size_t j = 0; j < Ny; j++) {
        for(size_t k = 0; k < Nz; k++) {
            size_t idx = (Nx-1) + j * Nx + k * Nx * Ny;
            solver.setBoundary(idx, BoundaryType::PressureOutlet);
            solver.setOutletPressure(idx, 1.0);
        }
    }
    
    // Create cylinder using IBM nodes
    std::cout << "Creating cylinder boundary...\n";
    constexpr int numIBMNodesCircum = 60;  // Number of points around circumference
    constexpr int numIBMNodesHeight = 30;  // Number of points along height
    
    for(int h = 0; h < numIBMNodesHeight; h++) {
        double z = cz - Nz/4 + (Nz/2) * h / (numIBMNodesHeight-1);
        for(int i = 0; i < numIBMNodesCircum; i++) {
            double theta = 2.0 * M_PI * i / numIBMNodesCircum;
            IBMNode<double, 3> node;
            node.position = {cx + D/2.0 * std::cos(theta),
                            cy + D/2.0 * std::sin(theta),
                            z};
            node.velocity = {0.0, 0.0, 0.0};  // Stationary cylinder
            node.force = {0.0, 0.0, 0.0};
            
            // Find near nodes and compute weights
            for(int dx = -2; dx <= 2; dx++) {
                for(int dy = -2; dy <= 2; dy++) {
                    for(int dz = -2; dz <= 2; dz++) {
                        int x = static_cast<int>(node.position[0]) + dx;
                        int y = static_cast<int>(node.position[1]) + dy;
                        int z = static_cast<int>(node.position[2]) + dz;
                        if(x >= 0 && x < Nx && y >= 0 && y < Ny && z >= 0 && z < Nz) {
                            size_t idx = x + y * Nx + z * Nx * Ny;
                            double dist = std::sqrt(std::pow(x - node.position[0], 2) +
                                                  std::pow(y - node.position[1], 2) +
                                                  std::pow(z - node.position[2], 2));
                            if(dist <= 2.0) {
                                node.nearNodes.push_back(idx);
                                node.weights.push_back(std::exp(-dist*dist/2.0));
                            }
                        }
                    }
                }
            }
            solver.addIBMNode(node);
        }
    }
    
    // Set nodes inside cylinder as inactive
    auto& nodes = mesh.getNodes();
    for(size_t k = 0; k < Nz; k++) {
        for(size_t j = 0; j < Ny; j++) {
            for(size_t i = 0; i < Nx; i++) {
                if(isInsideCylinder(i, j, k)) {
                    nodes[i + j*Nx + k*Nx*Ny].isActive = false;
                }
            }
        }
    }
    
    // Simulation parameters
    const int nSteps = 500000;
    const int saveInterval = 100;
    
    // Initialize visualization
    Visual<double, 3> visualizer;
    
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
                double velMag = std::sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
                maxVel = std::max(maxVel, velMag);
            }
            std::cout << ", Max velocity: " << maxVel << "\n";
            
            // Save visualization
            std::string filename = "cylinder_flow_3d_" + std::to_string(step/saveInterval);
            Visual<double, 3>::writeVTK(mesh, filename);
        }
    }
    
    // Calculate and print simulation time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "\nSimulation completed in " << duration.count() << " seconds\n";
    
    return 0;
}
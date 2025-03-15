#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include "../../src/PDEs/Mesh/MeshObj.hpp"

// Function to check if a point is inside a sphere
template <typename TNum>
bool isInsideSphere(const Point3D<TNum>& point, const Point3D<TNum>& center, TNum radius) {
    TNum dx = point.x - center.x;
    TNum dy = point.y - center.y;
    TNum dz = point.z - center.z;
    return (dx*dx + dy*dy + dz*dz) <= radius*radius;
}

// Function to initialize a velocity field with a parabolic profile
template <typename TNum>
void initializeParabolicFlow(MeshObj<TNum>& mesh, TNum max_velocity) {
    // Get domain dimensions
    TNum xmin = 0.0, xmax = 1.0;
    TNum ymin = 0.0, ymax = 1.0;
    TNum zmin = 0.0, zmax = 1.0;
    
    // Get all fluid cells
    auto fluid_cells = mesh.getFluidCells();
    
    for (int cellId : fluid_cells) {
        // Get cell center
        const auto& cell = mesh.getCell(cellId);
        TNum x = cell.center.x;
        TNum y = cell.center.y;
        TNum z = cell.center.z;
        
        // Calculate normalized y and z coordinates (0 to 1)
        TNum y_norm = (y - ymin) / (ymax - ymin);
        TNum z_norm = (z - zmin) / (zmax - zmin);
        
        // Parabolic profile: u = u_max * 4 * y_norm * (1 - y_norm) * 4 * z_norm * (1 - z_norm)
        TNum u = max_velocity * 4.0 * y_norm * (1.0 - y_norm) * 4.0 * z_norm * (1.0 - z_norm);
        
        // Set velocity (flow in x direction)
        mesh.setCellValues(cellId, u, 0.0, 0.0, 0.0);
    }
}

// Custom refinement criterion based on distance to sphere
template <typename TNum>
TNum distanceToSphereCriterion(const MeshCell<TNum>& cell, const Point3D<TNum>& sphere_center, TNum sphere_radius) {
    // Calculate distance from cell center to sphere surface
    TNum dx = cell.center.x - sphere_center.x;
    TNum dy = cell.center.y - sphere_center.y;
    TNum dz = cell.center.z - sphere_center.z;
    TNum distance = std::sqrt(dx*dx + dy*dy + dz*dz) - sphere_radius;
    
    // Return inverse of distance (higher value near sphere)
    return std::max(0.1, 1.0 / (std::abs(distance) + 0.1));
}

int main() {
    std::cout << "Starting Mesh-based CFD simulation..." << std::endl;
    
    // Simulation parameters
    using Real = double;
    Real xmin = 0.0, xmax = 2.0;
    Real ymin = 0.0, ymax = 1.0;
    Real zmin = 0.0, zmax = 1.0;
    int nx = 20, ny = 10, nz = 3;
    int max_level = 2;
    
    // Create mesh
    MeshObj<Real> mesh(xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz, max_level);
    
    // Define sphere as immersed boundary
    Point3D<Real> sphere_center(0.5, 0.5, 0.5);
    Real sphere_radius = 0.2;
    
    // Add sphere as immersed boundary
    mesh.addImmersedBoundary([sphere_center, sphere_radius](const Point3D<Real>& point) {
        return isInsideSphere(point, sphere_center, sphere_radius);
    });
    
    // Set custom refinement criterion based on distance to sphere
    mesh.setCustomRefinementCriterion(
        [sphere_center, sphere_radius](const MeshCell<Real>& cell) {
            return distanceToSphereCriterion(cell, sphere_center, sphere_radius);
        },
        0.5  // Refinement threshold
    );
    
    // Apply initial mesh refinement
    std::cout << "Refining mesh..." << std::endl;
    mesh.adaptMesh();
    mesh.adaptMesh();  // Apply multiple times for progressive refinement
    
    // Initialize flow field
    std::cout << "Initializing flow field..." << std::endl;
    Real max_velocity = 1.0;
    initializeParabolicFlow(mesh, max_velocity);
    
    // Export initial mesh
    std::cout << "Exporting initial mesh..." << std::endl;
    mesh.exportToVTK("flow_around_sphere_initial.vtk");
    
    // Simulation parameters
    int num_steps = 100;
    Real dt = 0.01;
    std::string output_prefix = "flow_around_sphere";
    int output_frequency = 10;
    
    // Run simulation
    std::cout << "Running simulation for " << num_steps << " steps..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    mesh.runSimulation(num_steps, dt, output_prefix, output_frequency);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << "Simulation completed in " << duration << " ms" << std::endl;
    std::cout << "Final mesh has " << mesh.getNumCells() << " cells and " << mesh.getNumNodes() << " nodes" << std::endl;
    
    // Print statistics
    std::cout << "Fluid cells: " << mesh.getFluidCells().size() << std::endl;
    std::cout << "Boundary cells: " << mesh.getBoundaryCells().size() << std::endl;
    std::cout << "Solid cells: " << mesh.getSolidCells().size() << std::endl;
    
    return 0;
}
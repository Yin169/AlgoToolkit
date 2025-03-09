#include "../../src/PDEs/FDM/NavierStoke.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

// Function to save velocity field to a file in VTK format
template <typename TNum>
void saveVTKFile(const NavierStokesSolver3D<TNum>& solver, const std::string& filename) {
    int nx = solver.getNx();
    int ny = solver.getNy();
    int nz = solver.getNz();
    TNum dx = solver.getDx();
    TNum dy = solver.getDy();
    TNum dz = solver.getDz();
    
    std::ofstream outFile(filename);
    outFile << "# vtk DataFile Version 3.0\n";
    outFile << "Jet Flow Simulation\n";
    outFile << "ASCII\n";
    outFile << "DATASET STRUCTURED_GRID\n";
    outFile << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
    outFile << "POINTS " << nx*ny*nz << " float\n";
    
    // Write grid points
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                outFile << i*dx << " " << j*dy << " " << k*dz << "\n";
            }
        }
    }
    
    // Write velocity vectors
    outFile << "POINT_DATA " << nx*ny*nz << "\n";
    outFile << "VECTORS velocity float\n";
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                outFile << solver.getU(i, j, k) << " " 
                        << solver.getV(i, j, k) << " " 
                        << solver.getW(i, j, k) << "\n";
            }
        }
    }
    
    // Write pressure field
    outFile << "SCALARS pressure float 1\n";
    outFile << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                outFile << solver.getP(i, j, k) << "\n";
            }
        }
    }
    
    outFile.close();
}

int main() {
    // Simulation parameters
    const int nx = 50;
    const int ny = 50;
    const int nz = 50;
    const double dx = 0.1;
    const double dy = 0.1;
    const double dz = 0.1;
    const double dt = 0.01;
    const double Re = 100.0;  // Reynolds number
    const double totalTime = 10.0;
    const int saveInterval = 20;  // Save every 20 steps
    
    // Create solver
    NavierStokesSolver3D<double> solver(nx, ny, nz, dx, dy, dz, dt, Re);
    
    // Define jet inlet boundary conditions
    // Jet enters from the x=0 plane in the center
    auto u_bc = [nx, ny, nz, dx, dy, dz](double x, double y, double z, double t) -> double {
        // Jet at the inlet (x=0)
        if (std::abs(x) < 1e-10) {
            // Circular jet in the center of the inlet plane
            double centerY = (ny/2) * dy;
            double centerZ = (nz/2) * dz;
            double jetRadius = std::min(ny, nz) * 0.1 * dy;  // 10% of domain size
            
            double distFromCenter = std::sqrt(std::pow(y - centerY, 2) + std::pow(z - centerZ, 2));
            
            if (distFromCenter <= jetRadius) {
                // Parabolic velocity profile
                return 1.0 * (1.0 - std::pow(distFromCenter/jetRadius, 2));
            }
        }
        // No-slip at other boundaries
        else if (std::abs(x - (nx-1)*dx) < 1e-10 || 
                 std::abs(y) < 1e-10 || std::abs(y - (ny-1)*dy) < 1e-10 ||
                 std::abs(z) < 1e-10 || std::abs(z - (nz-1)*dz) < 1e-10) {
            return 0.0;
        }
        
        return 0.0;
    };
    
    // Zero velocity in y and z directions at boundaries
    auto v_bc = [nx, ny, nz, dx, dy, dz](double x, double y, double z, double t) -> double {
        if (std::abs(x) < 1e-10 || std::abs(x - (nx-1)*dx) < 1e-10 || 
            std::abs(y) < 1e-10 || std::abs(y - (ny-1)*dy) < 1e-10 ||
            std::abs(z) < 1e-10 || std::abs(z - (nz-1)*dz) < 1e-10) {
            return 0.0;
        }
        return 0.0;
    };
    
    auto w_bc = [nx, ny, nz, dx, dy, dz](double x, double y, double z, double t) -> double {
        if (std::abs(x) < 1e-10 || std::abs(x - (nx-1)*dx) < 1e-10 || 
            std::abs(y) < 1e-10 || std::abs(y - (ny-1)*dy) < 1e-10 ||
            std::abs(z) < 1e-10 || std::abs(z - (nz-1)*dz) < 1e-10) {
            return 0.0;
        }
        return 0.0;
    };
    
    // Set boundary conditions
    solver.setBoundaryConditions(u_bc, v_bc, w_bc);
    
    // Set initial conditions (zero velocity everywhere)
    auto zero_init = [](double x, double y, double z) -> double { return 0.0; };
    solver.setInitialConditions(zero_init, zero_init, zero_init);
    
    // Run simulation
    int numSteps = static_cast<int>(totalTime / dt);
    std::cout << "Starting jet flow simulation for " << numSteps << " steps..." << std::endl;
    
    for (int step = 0; step < numSteps; step++) {
        double currentTime = step * dt;
        solver.step(currentTime);
        
        // Print progress
        if (step % 100 == 0) {
            double ke = solver.calculateKineticEnergy();
            std::cout << "Step " << step << "/" << numSteps 
                      << ", Time: " << currentTime 
                      << ", Kinetic Energy: " << ke << std::endl;
        }
        
        // Save results periodically
        if (step % saveInterval == 0) {
            std::string filename = "jetflow_" + std::to_string(step) + ".vtk";
            saveVTKFile(solver, filename);
        }
    }
    
    // Calculate and output vorticity at the end
    VectorObj<double> vort_x, vort_y, vort_z;
    solver.calculateVorticity(vort_x, vort_y, vort_z);
    
    // Save final state
    saveVTKFile(solver, "jetflow_final.vtk");
    
    std::cout << "Simulation completed. Results saved to VTK files." << std::endl;
    
    return 0;
}
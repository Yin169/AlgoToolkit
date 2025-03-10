#include "../../src/PDEs/FDM/NavierStoke.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <iomanip>

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
    
    // Calculate and write vorticity
    VectorObj<TNum> vort_x, vort_y, vort_z;
    solver.calculateVorticity(vort_x, vort_y, vort_z);
    
    // Vorticity vector field
    outFile << "VECTORS vorticity float\n";
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = i + j*nx + k*nx*ny;
                outFile << vort_x[idx] << " " << vort_y[idx] << " " << vort_z[idx] << "\n";
            }
        }
    }
    
    // Vorticity magnitude
    outFile << "SCALARS vorticity_magnitude float 1\n";
    outFile << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = i + j*nx + k*nx*ny;
                TNum magnitude = std::sqrt(vort_x[idx]*vort_x[idx] + 
                                          vort_y[idx]*vort_y[idx] + 
                                          vort_z[idx]*vort_z[idx]);
                outFile << magnitude << "\n";
            }
        }
    }
    
    // Calculate and write divergence (should be close to zero)
    VectorObj<TNum> div = solver.calculateDivergence();
    
    outFile << "SCALARS divergence float 1\n";
    outFile << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = i + j*nx + k*nx*ny;
                outFile << div[idx] << "\n";
            }
        }
    }
    
    outFile.close();
}

int main() {
    // Simulation parameters
    const int nx = 100;
    const int ny = 80;
    const int nz = 3;
    const double dx = 0.2;
    const double dy = 0.2;
    const double dz = 0.2;
    const double dt = 0.005;
    const double Re = 1000.0;  // Reynolds number
    const double totalTime = 20.0;
    const int saveInterval = 50;  // Save every 50 steps
    
    // Jet parameters
    const double jetRadius = 1.0;
    const double jetVelocity = 2.0;
    const double jetCenterY = ny * dy / 2.0;
    const double jetCenterZ = nz * dz / 2.0;
    
    // Create solver
    NavierStokesSolver3D<double> solver(nx, ny, nz, dx, dy, dz, dt, Re);
    
    // Define boundary conditions with convective outflow
    auto u_bc = [&solver, nx, ny, nz, dx, dy, dz, jetRadius, jetVelocity, jetCenterY, jetCenterZ]
               (double x, double y, double z, double t) -> double {
        // Jet inlet at x=0 (left boundary)
        if (std::abs(x) < 1e-10) {
            // Calculate distance from jet center
            double r = std::sqrt(std::pow(y - jetCenterY, 2) + std::pow(z - jetCenterZ, 2));
            
            if (r <= jetRadius) {
                // Smooth jet profile with cosine function for better stability
                double profile = 0.5 * (1.0 + std::cos(M_PI * r / jetRadius));
                return jetVelocity * profile;
            }
            return 0.0;
        }
        
        // Convective outflow at x=L (right boundary)
        if (std::abs(x - (nx-1)*dx) < 1e-10) {
            int i = nx-2;  // One cell before boundary
            int j = static_cast<int>(y/dy);
            int k = static_cast<int>(z/dz);
            
            // Ensure indices are within bounds
            j = std::max(0, std::min(j, ny-1));
            k = std::max(0, std::min(k, nz-1));
            
            return solver.getU(i, j, k);  // Zero gradient
        }
        
        // Free-slip at other boundaries
        return 0.0;
    };
    
    auto v_bc = [&solver, nx, ny, nz, dx, dy, dz]
               (double x, double y, double z, double t) -> double {
        // No flow through inlet
        if (std::abs(x) < 1e-10) {
            return 0.0;
        }
        
        // Convective outflow at x=L (right boundary)
        if (std::abs(x - (nx-1)*dx) < 1e-10) {
            int i = nx-2;  // One cell before boundary
            int j = static_cast<int>(y/dy);
            int k = static_cast<int>(z/dz);
            
            // Ensure indices are within bounds
            j = std::max(0, std::min(j, ny-1));
            k = std::max(0, std::min(k, nz-1));
            
            return solver.getV(i, j, k);  // Zero gradient
        }
        
        // Free-slip at y boundaries
        if (std::abs(y) < 1e-10 || std::abs(y - (ny-1)*dy) < 1e-10) {
            int i = static_cast<int>(x/dx);
            int k = static_cast<int>(z/dz);
            
            // Ensure indices are within bounds
            i = std::max(0, std::min(i, nx-1));
            k = std::max(0, std::min(k, nz-1));
            
            return 0.0;  // No flow through boundary
        }
        
        // Free-slip at z boundaries
        if (std::abs(z) < 1e-10 || std::abs(z - (nz-1)*dz) < 1e-10) {
            int i = static_cast<int>(x/dx);
            int j = static_cast<int>(y/dy);
            
            // Ensure indices are within bounds
            i = std::max(0, std::min(i, nx-1));
            j = std::max(0, std::min(j, ny-1));
            
            return 0.0;  // No flow through boundary
        }
        
        return 0.0;
    };
    
    auto w_bc = [&solver, nx, ny, nz, dx, dy, dz]
               (double x, double y, double z, double t) -> double {
        // No flow through inlet
        if (std::abs(x) < 1e-10) {
            return 0.0;
        }
        
        // Convective outflow at x=L (right boundary)
        if (std::abs(x - (nx-1)*dx) < 1e-10) {
            int i = nx-2;  // One cell before boundary
            int j = static_cast<int>(y/dy);
            int k = static_cast<int>(z/dz);
            
            // Ensure indices are within bounds
            j = std::max(0, std::min(j, ny-1));
            k = std::max(0, std::min(k, nz-1));
            
            return solver.getW(i, j, k);  // Zero gradient
        }
        
        // Free-slip at y boundaries
        if (std::abs(y) < 1e-10 || std::abs(y - (ny-1)*dy) < 1e-10) {
            int i = static_cast<int>(x/dx);
            int k = static_cast<int>(z/dz);
            
            // Ensure indices are within bounds
            i = std::max(0, std::min(i, nx-1));
            k = std::max(0, std::min(k, nz-1));
            
            return 0.0;  // No flow through boundary
        }
        
        // Free-slip at z boundaries
        if (std::abs(z) < 1e-10 || std::abs(z - (nz-1)*dz) < 1e-10) {
            int i = static_cast<int>(x/dx);
            int j = static_cast<int>(y/dy);
            
            // Ensure indices are within bounds
            i = std::max(0, std::min(i, nx-1));
            j = std::max(0, std::min(j, ny-1));
            
            return 0.0;  // No flow through boundary
        }
        
        return 0.0;
    };
    
    // Set boundary conditions
    solver.setBoundaryConditions(u_bc, v_bc, w_bc);
    
    // Set initial conditions (small random perturbation for better vortex development)
    auto u_init = [jetVelocity, jetRadius, jetCenterY, jetCenterZ, dx, dy, dz]
                 (double x, double y, double z) -> double {
        // Small initial velocity in the domain
        double r = std::sqrt(std::pow(y - jetCenterY, 2) + std::pow(z - jetCenterZ, 2));
        
        if (x < 2.0*dx && r <= jetRadius) {
            // Smooth jet profile with cosine function
            double profile = 0.5 * (1.0 + std::cos(M_PI * r / jetRadius));
            return jetVelocity * profile;
        }
        
        // Small random perturbation elsewhere
        return 0.01 * jetVelocity * (2.0 * rand() / RAND_MAX - 1.0);
    };
    
    auto v_init = [jetVelocity](double x, double y, double z) -> double {
        // Small random perturbation
        return 0.01 * jetVelocity * (2.0 * rand() / RAND_MAX - 1.0);
    };
    
    auto w_init = [jetVelocity](double x, double y, double z) -> double {
        // Small random perturbation
        return 0.01 * jetVelocity * (2.0 * rand() / RAND_MAX - 1.0);
    };
    
    // Set initial conditions
    solver.setInitialConditions(u_init, v_init, w_init);
    
    // Run simulation
    int numSteps = static_cast<int>(totalTime / dt);
    std::cout << "Starting jet flow simulation for " << numSteps << " steps..." << std::endl;
    
    // Create output directory
    system("mkdir -p jet_flow_results");
    
    for (int step = 0; step < numSteps; step++) {
        double currentTime = step * dt;
        solver.step(currentTime);
        
        // Print progress and energy
        if (step % 100 == 0) {
            double ke = solver.calculateKineticEnergy();
            std::cout << "Step " << step << "/" << numSteps 
                      << ", Time: " << std::fixed << std::setprecision(3) << currentTime 
                      << ", Kinetic Energy: " << std::scientific << std::setprecision(6) << ke << std::endl;
            
            // Check for divergence (optional stability check)
            VectorObj<double> div = solver.calculateDivergence();
            double max_div = 0.0;
            for (int i = 0; i < div.size(); i++) {
                max_div = std::max(max_div, std::abs(div[i]));
            }
            std::cout << "  Max Divergence: " << std::scientific << max_div << std::endl;
        }
        
        // Save results periodically
        if (step % saveInterval == 0) {
            std::string filename = "jet_flow_results/jetflow_" + std::to_string(step/saveInterval) + ".vtk";
            saveVTKFile(solver, filename);
        }
    }
    
    // Save final state
    saveVTKFile(solver, "jet_flow_results/jetflow_final.vtk");
    
    std::cout << "Simulation completed. Results saved to jet_flow_results directory." << std::endl;
    
    return 0;
}
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
    
    // Get velocity and pressure fields
    const VectorObj<TNum>& u = solver.getU();
    const VectorObj<TNum>& v = solver.getV();
    const VectorObj<TNum>& w = solver.getW();
    const VectorObj<TNum>& p = solver.getP();
    
    // Helper function to get index in the flattened array
    auto idx = [nx, ny](int i, int j, int k) -> int {
        return i + j * nx + k * nx * ny;
    };
    
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
                int index = idx(i, j, k);
                outFile << u[index] << " " << v[index] << " " << w[index] << "\n";
            }
        }
    }
    
    // Write pressure field
    outFile << "SCALARS pressure float 1\n";
    outFile << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int index = idx(i, j, k);
                outFile << p[index] << "\n";
            }
        }
    }
    
    // Calculate and write vorticity
    auto vorticity = solver.calculateVorticity();
    VectorObj<TNum>& vort_x = std::get<0>(vorticity);
    VectorObj<TNum>& vort_y = std::get<1>(vorticity);
    VectorObj<TNum>& vort_z = std::get<2>(vorticity);
    
    // Vorticity vector field
    outFile << "VECTORS vorticity float\n";
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int index = idx(i, j, k);
                outFile << vort_x[index] << " " << vort_y[index] << " " << vort_z[index] << "\n";
            }
        }
    }
    
    // Vorticity magnitude
    outFile << "SCALARS vorticity_magnitude float 1\n";
    outFile << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int index = idx(i, j, k);
                TNum magnitude = std::sqrt(vort_x[index]*vort_x[index] + 
                                          vort_y[index]*vort_y[index] + 
                                          vort_z[index]*vort_z[index]);
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
                int index = idx(i, j, k);
                outFile << div[index] << "\n";
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
    
    // Helper function to get index in the flattened array
    auto idx = [nx, ny](int i, int j, int k) -> int {
        return i + j * nx + k * nx * ny;
    };
    
    // Define boundary conditions with convective outflow
    auto u_bc = [jetRadius, jetVelocity, jetCenterY, jetCenterZ]
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
        
        // For other boundaries, we'll use zero gradient in the solver's implementation
        return 0.0;
    };
    
    auto v_bc = [](double x, double y, double z, double t) -> double {
        // No flow through boundaries
        return 0.0;
    };
    
    auto w_bc = [](double x, double y, double z, double t) -> double {
        // No flow through boundaries
        return 0.0;
    };
    
    // Set boundary conditions
    solver.setBoundaryConditions(u_bc, v_bc, w_bc);
    
    // Initialize velocity fields
    int n = nx * ny * nz;
    VectorObj<double> u0(n, 0.0);
    VectorObj<double> v0(n, 0.0);
    VectorObj<double> w0(n, 0.0);
    
    // Set initial conditions with small random perturbation
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int index = idx(i, j, k);
                double x = i * dx;
                double y = j * dy;
                double z = k * dz;
                
                // Jet profile near inlet
                double r = std::sqrt(std::pow(y - jetCenterY, 2) + std::pow(z - jetCenterZ, 2));
                
                if (x < 2.0*dx && r <= jetRadius) {
                    // Smooth jet profile with cosine function
                    double profile = 0.5 * (1.0 + std::cos(M_PI * r / jetRadius));
                    u0[index] = jetVelocity * profile;
                } else {
                    // Small random perturbation elsewhere
                    u0[index] = 0.01 * jetVelocity * (2.0 * rand() / RAND_MAX - 1.0);
                }
                
                // Small random perturbation for v and w
                v0[index] = 0.01 * jetVelocity * (2.0 * rand() / RAND_MAX - 1.0);
                w0[index] = 0.01 * jetVelocity * (2.0 * rand() / RAND_MAX - 1.0);
            }
        }
    }
    
    // Set initial conditions
    solver.setInitialConditions(u0, v0, w0);
    
    // Set advection scheme to QUICK for better accuracy
    solver.setAdvectionScheme(AdvectionScheme::QUICK);
    
    // Enable adaptive time stepping for stability
    solver.enableAdaptiveTimeStep(0.5);
    
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
            
            // Also print current dt if adaptive time stepping is enabled
            std::cout << "  Current dt: " << std::scientific << solver.getDt() << std::endl;
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
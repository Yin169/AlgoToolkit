#include "../../src/PDEs/FDM/NavierStoke.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <iomanip>

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
    const int nx = 64;
    const int ny = 64;
    const int nz = 64;  // Using 3 layers for quasi-2D simulation
    const double L = 1.0;  // Domain size
    const double dx = L / (nx - 1);
    const double dy = L / (ny - 1);
    const double dz = L / (nz - 1);
    const double dt = 0.0005;  // Reduced initial time step for stability
    const double Re = 100.0;  // Reynolds number
    const double U_lid = 10.0;  // Lid velocity
	const double totalTime = 20.0;  // Total simulation time
	const int saveInterval = 50;  // Save every 50 steps
    
    // Create solver
    NavierStokesSolver3D<double> solver(nx, ny, nz, dx, dy, dz, dt, Re);
    
    // Set boundary conditions for lid-driven cavity
    // Top lid moves with constant velocity U_lid in x-direction
    auto u_bc = [U_lid, ny, dy](double x, double y, double z, double t) -> double {
        if (std::abs(y - (ny-1)*dy) < 1e-10) {
            return U_lid;  // Top boundary (lid)
        }
        return 0.0;  // All other boundaries
    };
    
    auto v_bc = [](double x, double y, double z, double t) -> double {
        return 0.0;  // Zero velocity in y-direction on all boundaries
    };
    
    auto w_bc = [](double x, double y, double z, double t) -> double {
        return 0.0;  // Zero velocity in z-direction on all boundaries
    };
    
    solver.setBoundaryConditions(u_bc, v_bc, w_bc);
    
    // Set time integration method and advection scheme
    solver.setTimeIntegrationMethod(TimeIntegration::EULER);  // Use simpler Euler method for stability
    solver.setAdvectionScheme(AdvectionScheme::UPWIND);  // Use more stable upwind scheme
    
    
    std::cout << "Starting lid-driven cavity simulation..." << std::endl;
    std::cout << "Reynolds number: " << Re << std::endl;
    std::cout << "Grid size: " << nx << "x" << ny << "x" << nz << std::endl;

	int numSteps = static_cast<int>(totalTime / dt);
    std::cout << "Starting jet flow simulation for " << numSteps << " steps..." << std::endl;
	
	system("mkdir -p output");

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
            std::string filename = "output/cavity_" + std::to_string(step) + ".vtk"; 
            saveVTKFile(solver, filename);
        }
    }
    
    return 0;
}
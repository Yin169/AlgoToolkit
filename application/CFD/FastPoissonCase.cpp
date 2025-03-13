#include <iostream>
#include <iomanip>
#include <cmath>
#include "FastPoisson.hpp"

// Function to calculate the analytical solution for comparison
double analyticalSolution(double x, double y, double z) {
    // Solution to ∇²u = -sin(πx)sin(πy)sin(πz) is u = sin(πx)sin(πy)sin(πz)/(3π²)
    return sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z) / (3 * M_PI * M_PI);
}

// Function to calculate the right-hand side of the Poisson equation
double sourceFunction(double x, double y, double z) {
    // f = -∇²u = sin(πx)sin(πy)sin(πz)
    return -sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
}

int main() {
    // Domain parameters
    const int nx = 64;
    const int ny = 64;
    const int nz = 64;
    const double Lx = 1.0;
    const double Ly = 1.0;
    const double Lz = 1.0;
    
    // Create the Poisson solver
    FastPoisson3D<double> solver(nx, ny, nz, Lx, Ly, Lz);
    solver.setRHS(sourceFunction);
    solver.setDirichletBC(analyticalSolution);
    
    // Grid spacing
    const double dx = solver.getDx();
    const double dy = solver.getDy();
    const double dz = solver.getDz();
    
    // Solve the Poisson equation
    std::cout << "Solving the Poisson equation..." << std::endl;
    VectorObj<double> solution = solver.solve();
    std::cout << "Solution completed." << std::endl;
    
    // Calculate error compared to analytical solution
    double maxError = 0.0;
    double l2Error = 0.0;
    
    for (int i = 0; i < nx; i++) {
        double x = i * dx;
        for (int j = 0; j < ny; j++) {
            double y = j * dy;
            for (int k = 0; k < nz; k++) {
                double z = k * dz;
                int idx = solver.index(i, j, k);
                
                double numerical = solution[idx];
                double analytical = analyticalSolution(x, y, z);
                double error = std::abs(numerical - analytical);
                
                maxError = std::max(maxError, error);
                l2Error += error * error;
            }
        }
    }
    
    l2Error = std::sqrt(l2Error / (nx * ny * nz));
    
    // Output results
    std::cout << "Results:" << std::endl;
    std::cout << "  Grid size: " << nx << " x " << ny << " x " << nz << std::endl;
    std::cout << "  Maximum error: " << std::scientific << std::setprecision(6) << maxError << std::endl;
    std::cout << "  L2 error: " << std::scientific << std::setprecision(6) << l2Error << std::endl;
    
    // Output a slice of the solution for visualization
    std::cout << "\nSolution values at z = " << Lz/2 << ":" << std::endl;
    int k_mid = nz / 2;
    
    for (int i = 0; i < std::min(10, nx); i++) {
        for (int j = 0; j < std::min(10, ny); j++) {
            std::cout << std::fixed << std::setprecision(6) << std::setw(12) 
                      << solution[solver.index(i, j, k_mid)] << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}
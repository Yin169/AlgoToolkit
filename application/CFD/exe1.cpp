#include "../../src/PDEs/FDM/Laplacian.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <functional>
#include <fstream>
#include <string>
#include <chrono>

// Analytical solution: u(x,y,z) = sin(πx)sin(πy)sin(πz)
// This gives us f(x,y,z) = -∇²u = 3π²sin(πx)sin(πy)sin(πz)

double analyticalSolution(double x, double y, double z) {
    return sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
}

double rhsFunction(double x, double y, double z) {
    // -∇²u = 3π²sin(πx)sin(πy)sin(πz)
    return -3.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
}

double boundaryCondition(double x, double y, double z) {
    // Dirichlet boundary condition: u = sin(πx)sin(πy)sin(πz) on the boundary
    return analyticalSolution(x, y, z);
}

// Function to calculate L2 error between numerical and analytical solutions
double calculateL2Error(const Poisson3DSolver<double>& solver) {
    double error = 0.0;
    double totalPoints = 0.0;
    
    int nx = solver.getNx();
    int ny = solver.getNy();
    int nz = solver.getNz();
    double hx = solver.getDx();
    double hy = solver.getDy();
    double hz = solver.getDz();
    
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double x = i * hx;
                double y = j * hy;
                double z = k * hz;
                
                double numerical = solver.getSolution(i, j, k);
                double analytical = analyticalSolution(x, y, z);
                
                error += (numerical - analytical) * (numerical - analytical);
                totalPoints += 1.0;
            }
        }
    }
    
    return sqrt(error / totalPoints);
}

// Function to calculate maximum error between numerical and analytical solutions
double calculateMaxError(const Poisson3DSolver<double>& solver) {
    double maxError = 0.0;
    
    int nx = solver.getNx();
    int ny = solver.getNy();
    int nz = solver.getNz();
    double hx = solver.getDx();
    double hy = solver.getDy();
    double hz = solver.getDz();
    
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double x = i * hx;
                double y = j * hy;
                double z = k * hz;
                
                double numerical = solver.getSolution(i, j, k);
                double analytical = analyticalSolution(x, y, z);
                
                double error = std::abs(numerical - analytical);
                maxError = std::max(maxError, error);
            }
        }
    }
    
    return maxError;
}

void exportToVTK(const std::string& filename, const Poisson3DSolver<double>& solver) {
    int nx = solver.getNx();
    int ny = solver.getNy();
    int nz = solver.getNz();
    double dx = solver.getDx();
    double dy = solver.getDy();
    double dz = solver.getDz();
    
    // Get solution field
    const VectorObj<double>& u = solver.getSolution();
    
    // Helper function to get index in the flattened array
    auto idx = [nx, ny](int i, int j, int k) -> int {
        return i + j * nx + k * nx * ny;
    };
    
    std::ofstream outFile(filename);
    outFile << "# vtk DataFile Version 3.0\n";
    outFile << "Poisson Simulation\n";
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
    
    // Write data header
    outFile << "POINT_DATA " << nx*ny*nz << "\n";
    
    // Write numerical solution
    outFile << "SCALARS numerical_solution float 1\n";
    outFile << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                outFile << u[idx(i, j, k)] << "\n";
            }
        }
    }
    
    // Write analytical solution for comparison
    outFile << "SCALARS analytical_solution float 1\n";
    outFile << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double x = i * dx;
                double y = j * dy;
                double z = k * dz;
                outFile << analyticalSolution(x, y, z) << "\n";
            }
        }
    }
    
    // Write error field
    outFile << "SCALARS error float 1\n";
    outFile << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double x = i * dx;
                double y = j * dy;
                double z = k * dz;
                double numerical = u[idx(i, j, k)];
                double analytical = analyticalSolution(x, y, z);
                outFile << std::abs(numerical - analytical) << "\n";
            }
        }
    }
    
    outFile.close();
    std::cout << "Solution exported to " << filename << " (ParaView compatible format)" << std::endl;
}

int main() {
    // Problem parameters
    int nx = 128;  // Number of grid points in x direction
    int ny = 128;  // Number of grid points in y direction
    int nz = 128;  // Number of grid points in z direction
    double lx = 1.0;  // Domain size in x direction
    double ly = 1.0;  // Domain size in y direction
    double lz = 1.0;  // Domain size in z direction
    
    std::cout << "=== 3D Poisson Equation Solver ===" << std::endl;
    std::cout << "Creating 3D Poisson solver with " << nx << "x" << ny << "x" << nz << " grid points..." << std::endl;
    
    // Create the Poisson solver
    auto startTime = std::chrono::high_resolution_clock::now();
    
    Poisson3DSolver<double> solver(nx, ny, nz, lx, ly, lz);
    
    // Set the right-hand side function
    std::cout << "Setting right-hand side function..." << std::endl;
    solver.setRHS(rhsFunction);
    
    // Set Dirichlet boundary conditions
    std::cout << "Setting Dirichlet boundary conditions..." << std::endl;
    solver.setDirichletBC(boundaryCondition);
    
    // Solve the Poisson equation
    std::cout << "Solving the Poisson equation..." << std::endl;
    solver.solve(1000, 1e-8);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    std::cout << "Execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
    
    // Calculate and print the L2 error
    double l2Error = calculateL2Error(solver);
    double maxError = calculateMaxError(solver);
    std::cout << "L2 error: " << std::scientific << std::setprecision(6) << l2Error << std::endl;
    std::cout << "Max error: " << std::scientific << std::setprecision(6) << maxError << std::endl;
    
    // Save solution to VTK file for visualization
    exportToVTK("solution_poisson.vtk", solver);
    
    // Print solution at some sample points
    std::cout << "\nSolution at sample points:" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    
    // Center of the domain
    double xCenter = 0.5, yCenter = 0.5, zCenter = 0.5;
    int iCenter = static_cast<int>(xCenter / solver.getDx());
    int jCenter = static_cast<int>(yCenter / solver.getDy());
    int kCenter = static_cast<int>(zCenter / solver.getDz());
    
    std::cout << "At (0.5, 0.5, 0.5): " << std::endl;
    std::cout << "  Numerical: " << solver.getSolution(iCenter, jCenter, kCenter) << std::endl;
    std::cout << "  Analytical: " << analyticalSolution(xCenter, yCenter, zCenter) << std::endl;
    
    // Quarter of the domain
    double xQuarter = 0.25, yQuarter = 0.25, zQuarter = 0.25;
    int iQuarter = static_cast<int>(xQuarter / solver.getDx());
    int jQuarter = static_cast<int>(yQuarter / solver.getDy());
    int kQuarter = static_cast<int>(zQuarter / solver.getDz());
    
    std::cout << "At (0.25, 0.25, 0.25): " << std::endl;
    std::cout << "  Numerical: " << solver.getSolution(iQuarter, jQuarter, kQuarter) << std::endl;
    std::cout << "  Analytical: " << analyticalSolution(xQuarter, yQuarter, zQuarter) << std::endl;
    
    
    return 0;
}
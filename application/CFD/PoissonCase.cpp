#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <fstream>
#include "../../src/PDEs/FDM/Poisson.hpp"

// Function to calculate the L2 error between numerical and analytical solutions
template <typename T>
double calculateL2Error(const FDM::PoissonSolver3D<T>& solver, 
                        const std::function<T(double, double, double)>& exactSolution) {
    double error = 0.0;
    int nx = solver.getNx();
    int ny = solver.getNy();
    int nz = solver.getNz();
    
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double x, y, z;
                solver.getCoordinates(i, j, k, x, y, z);
                
                double exact = exactSolution(x, y, z);
                double numerical = solver.getSolution(i, j, k);
                
                error += std::pow(exact - numerical, 2);
            }
        }
    }
    
    return std::sqrt(error / (nx * ny * nz));
}

// Function to save the solution to a VTK file for visualization
template <typename T>
void saveToVTK(const FDM::PoissonSolver3D<T>& solver, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    int nx = solver.getNx();
    int ny = solver.getNy();
    int nz = solver.getNz();
    
    // VTK file header
    file << "# vtk DataFile Version 3.0\n";
    file << "3D Poisson Solution\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
    file << "ORIGIN 0 0 0\n";
    file << "SPACING " << solver.getHx() << " " << solver.getHy() << " " << solver.getHz() << "\n";
    file << "POINT_DATA " << nx * ny * nz << "\n";
    file << "SCALARS solution float 1\n";
    file << "LOOKUP_TABLE default\n";
    
    // Write the solution data
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                file << solver.getSolution(i, j, k) << "\n";
            }
        }
    }
    
    file.close();
    std::cout << "Solution saved to " << filename << std::endl;
}

int main() {
    // Problem parameters
    const int nx = 65;  // Grid points in x-direction
    const int ny = 65;  // Grid points in y-direction
    const int nz = 65;  // Grid points in z-direction
    const double lx = 1.0;  // Domain length in x-direction
    const double ly = 1.0;  // Domain length in y-direction
    const double lz = 1.0;  // Domain length in z-direction
    
    std::cout << "Setting up 3D Poisson problem with " << nx << "x" << ny << "x" << nz << " grid points..." << std::endl;
    
    // Create the Poisson solver
    FDM::PoissonSolver3D<double> solver(nx, ny, nz, lx, ly, lz);
    
    // Define the analytical solution: u(x,y,z) = sin(πx)sin(πy)sin(πz)
    std::function<double(double, double, double)> exactSolution = 
        [](double x, double y, double z) -> double {
            return std::sin(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z);
        };
    
    // The source term f = -∇²u = 3π²sin(πx)sin(πy)sin(πz)
    auto sourceFunction = [](double x, double y, double z) -> double {
        return -3.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z);
    };
    
    
    // Boundary conditions (Dirichlet): u = exactSolution on the boundary
    auto boundaryFunction = exactSolution;
    
    // Now solve using SOR method for comparison
    std::cout << "\nSolving using SOR method..." << std::endl;
    
    // Create a new solver for SOR
    FDM::PoissonSolver3D<double> solverSOR(nx, ny, nz, lx, ly, lz);
    solverSOR.solve(sourceFunction, boundaryFunction, 5000, 1.5);
    
    
    // Calculate the L2 error for SOR
    auto l2Error = calculateL2Error(solverSOR, exactSolution);
    std::cout << "L2 error: " << l2Error << std::endl;
    
    // Save the SOR solution to a VTK file
    saveToVTK(solverSOR, "poisson_solution_sor.vtk");
    
    
    return 0;
}
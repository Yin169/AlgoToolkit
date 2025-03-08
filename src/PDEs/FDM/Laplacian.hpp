#ifndef LAPLACIAN_HPP
#define LAPLACIAN_HPP

#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <fstream> 
#include "../../Obj/SparseObj.hpp"
#include "../../Obj/VectorObj.hpp"
#include "../../LinearAlgebra/Solver/IterSolver.hpp"

template <typename TObj>
class Laplacian3DFDM {
private:
    int nx, ny, nz;           // Number of grid points in each dimension
    double hx, hy, hz;        // Grid spacing in each dimension
    double tol;               // Convergence tolerance
    int max_iter;             // Maximum iterations
    
    // Maps 3D index (i,j,k) to 1D index
    inline int idx(int i, int j, int k) const {
        return i + j * nx + k * nx * ny;
    }
    
    // Boundary condition function type
    using BoundaryConditionFunc = std::function<TObj(double, double, double)>;
    using SourceTermFunc = std::function<TObj(double, double, double)>;
public:
    Laplacian3DFDM(int nx_, int ny_, int nz_, 
                  double xmin, double xmax, 
                  double ymin, double ymax, 
                  double zmin, double zmax,
                  double tol_ = 1e-6, 
                  int max_iter_ = 10000) 
        : nx(nx_), ny(ny_), nz(nz_), 
          hx((xmax - xmin) / (nx - 1)), 
          hy((ymax - ymin) / (ny - 1)), 
          hz((zmax - zmin) / (nz - 1)),
          tol(tol_), max_iter(max_iter_) {}
    
    // Build the coefficient matrix for the Laplacian operator
    SparseMatrixCSC<TObj> buildLaplacianMatrix() const {
        const int total_points = nx * ny * nz;
        SparseMatrixCSC<TObj> A(total_points, total_points);
        
        // Coefficients for the finite difference stencil
        const double cx = 1.0 / (hx * hx);
        const double cy = 1.0 / (hy * hy);
        const double cz = 1.0 / (hz * hz);
        const double cc = -2.0 * (cx + cy + cz);  // Center coefficient
        
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int index = idx(i, j, k);
                    
                    // For boundary points, set diagonal to 1 (Dirichlet BC will be applied later)
                    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) {
                        A.addValue(index, index, 1.0);
                    } else {
                        // Interior points: apply the 7-point stencil
                        A.addValue(index, index, cc);
                        
                        // x-direction neighbors
                        A.addValue(index, idx(i-1, j, k), cx);
                        A.addValue(index, idx(i+1, j, k), cx);
                        
                        // y-direction neighbors
                        A.addValue(index, idx(i, j-1, k), cy);
                        A.addValue(index, idx(i, j+1, k), cy);
                        
                        // z-direction neighbors
                        A.addValue(index, idx(i, j, k-1), cz);
                        A.addValue(index, idx(i, j, k+1), cz);
                    }
                }
            }
        }
        
        A.finalize();
        return A;
    }
 
    void applySourceTerm(VectorObj<TObj>& b, 
                        const SourceTermFunc& sourceFunc,
                        double xmin, double ymin, double zmin) const {
        for (int k = 0; k < nz; ++k) {
            double z = zmin + k * hz;
            for (int j = 0; j < ny; ++j) {
                double y = ymin + j * hy;
                for (int i = 0; i < nx; ++i) {
                    double x = xmin + i * hx;
                        
                    // Skip boundary points (they're handled by Dirichlet BC)
                    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) {continue;}
                    // For interior points, apply the source term
                     b[idx(i, j, k)] = sourceFunc(x, y, z);
                }
            }
        }
    }
    
    // Apply Dirichlet boundary conditions
    void applyDirichletBC(VectorObj<TObj>& b, 
                         const BoundaryConditionFunc& bcFunc,
                         double xmin, double ymin, double zmin) const {
        for (int k = 0; k < nz; ++k) {
            double z = zmin + k * hz;
            for (int j = 0; j < ny; ++j) {
                double y = ymin + j * hy;
                for (int i = 0; i < nx; ++i) {
                    double x = xmin + i * hx;
                    
                    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) {
                        b[idx(i, j, k)] = bcFunc(x, y, z);
                    }
                }
            }
        }
    }
    
    VectorObj<TObj> solve(const BoundaryConditionFunc& bcFunc,
                         const SourceTermFunc& sourceFunc,
                         double xmin, double ymin, double zmin,
                         double omega = 1.0) const {
        const int total_points = nx * ny * nz;
        
        // Build the coefficient matrix
        SparseMatrixCSC<TObj> A = buildLaplacianMatrix();
        
        // Initialize the right-hand side vector (initially zero)
        VectorObj<TObj> b(total_points, 0.0);

        // Apply source term for interior points
        applySourceTerm(b, sourceFunc, xmin, ymin, zmin);
        // Apply boundary conditions
        applyDirichletBC(b, bcFunc, xmin, ymin, zmin);
        
        // Initialize solution vector with boundary conditions
        VectorObj<TObj> x(b.size(), 0.0);
        
        // Create and run the solver
        SOR<TObj, SparseMatrixCSC<TObj>, VectorObj<TObj>> solver(A, b, max_iter, omega);
        solver.solve(x);
        
        return x;
    }
    
    // Extract solution at a specific point
    TObj getSolutionAt(const VectorObj<TObj>& solution, int i, int j, int k) const {
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) {
            throw std::out_of_range("Grid indices out of range");
        }
        return solution[idx(i, j, k)];
    }
    
    // Get physical coordinates from grid indices
    void getCoordinates(int i, int j, int k, double xmin, double ymin, double zmin,
                       double& x, double& y, double& z) const {
        x = xmin + i * hx;
        y = ymin + j * hy;
        z = zmin + k * hz;
    }
    
    // Export solution to a VTK file for visualization
    void exportToVTK(const VectorObj<TObj>& solution, 
                    const std::string& filename,
                    double xmin, double ymin, double zmin) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }
        
        // VTK file header
        file << "# vtk DataFile Version 3.0\n";
        file << "3D Laplacian Solution\n";
        file << "ASCII\n";
        file << "DATASET STRUCTURED_POINTS\n";
        file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
        file << "ORIGIN " << xmin << " " << ymin << " " << zmin << "\n";
        file << "SPACING " << hx << " " << hy << " " << hz << "\n";
        file << "POINT_DATA " << nx * ny * nz << "\n";
        file << "SCALARS solution float 1\n";
        file << "LOOKUP_TABLE default\n";
        
        // Write solution values
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    file << solution[idx(i, j, k)] << "\n";
                }
            }
        }
        
        file.close();
    }
};

#endif // LAPLACIAN_HPP
#ifndef POISSON_HPP
#define POISSON_HPP

#include "../../Obj/SparseObj.hpp"
#include "../../Obj/VectorObj.hpp"
#include "../../LinearAlgebra/Solver/IterSolver.hpp"
#include <functional>
#include <cmath>
#include <iostream>
#include <vector>

namespace FDM {

template <typename TNum>
class PoissonSolver3D {
private:
    int nx, ny, nz;           // Number of grid points in each dimension
    double hx, hy, hz;        // Grid spacing in each dimension
    double lx, ly, lz;        // Domain size in each dimension
    SparseMatrixCSC<TNum> A;  // System matrix
    VectorObj<TNum> b;        // Right-hand side vector
    VectorObj<TNum> u;        // Solution vector
    
    // Maps 3D index (i,j,k) to 1D index
    inline int idx(int i, int j, int k) const {
        return i + j * nx + k * nx * ny;
    }
    
    // Builds the system matrix for the 3D Poisson equation
    void buildSystemMatrix() {
        const int n = nx * ny * nz;
        A = SparseMatrixCSC<TNum>(n, n);
        
        // Coefficients for the finite difference stencil
        const double cx = 1.0 / (hx * hx);
        const double cy = 1.0 / (hy * hy);
        const double cz = 1.0 / (hz * hz);
        const double cc = -2.0 * (cx + cy + cz); // Center coefficient
        
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int index = idx(i, j, k);
                    
                    // Center point
                    A.addValue(index, index, cc);
                    
                    // x-direction neighbors
                    if (i > 0) A.addValue(index, idx(i-1, j, k), cx);
                    if (i < nx-1) A.addValue(index, idx(i+1, j, k), cx);
                    
                    // y-direction neighbors
                    if (j > 0) A.addValue(index, idx(i, j-1, k), cy);
                    if (j < ny-1) A.addValue(index, idx(i, j+1, k), cy);
                    
                    // z-direction neighbors
                    if (k > 0) A.addValue(index, idx(i, j, k-1), cz);
                    if (k < nz-1) A.addValue(index, idx(i, j, k+1), cz);
                }
            }
        }
        
        A.finalize();
    }
    
    // Applies Dirichlet boundary conditions
    void applyDirichletBC(const std::function<TNum(double, double, double)>& boundaryFunc) {
        const int n = nx * ny * nz;
        
        // Reset the right-hand side vector
        b.zero();
        
        // Apply boundary conditions
        for (int k = 0; k < nz; ++k) {
            double z = k * hz;
            for (int j = 0; j < ny; ++j) {
                double y = j * hy;
                for (int i = 0; i < nx; ++i) {
                    double x = i * hx;
                    const int index = idx(i, j, k);
                    
                    // Check if this is a boundary point
                    bool isBoundary = (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1);
                    
                    if (isBoundary) {
                        // Set diagonal to 1
                        for (int idx = A.col_ptr[index]; idx < A.col_ptr[index+1]; ++idx) {
                            A.values[idx] = (A.row_indices[idx] == index) ? 1.0 : 0.0;
                        }
                        
                        // Set boundary value
                        b[index] = boundaryFunc(x, y, z);
                        u[index] = boundaryFunc(x, y, z);
                    }
                }
            }
        }
    }

public:
    PoissonSolver3D(int nx_, int ny_, int nz_, double lx_, double ly_, double lz_)
        : nx(nx_), ny(ny_), nz(nz_), 
          lx(lx_), ly(ly_), lz(lz_),
          hx(lx_ / (nx_ - 1)), hy(ly_ / (ny_ - 1)), hz(lz_ / (nz_ - 1)),
          b(nx_ * ny_ * nz_), u(nx_ * ny_ * nz_) {
        buildSystemMatrix();
    }
    
    // Solves the Poisson equation: ∇²u = f with Dirichlet boundary conditions
    // Alternative solver using SOR method
    void solve(const std::function<TNum(double, double, double)>& sourceFunc,
                  const std::function<TNum(double, double, double)>& boundaryFunc,
                  int maxIter = 1000, TNum omega = 1.5) {
        
        const int n = nx * ny * nz;
        
        // Set up the right-hand side vector with the source function
        for (int k = 1; k < nz-1; ++k) {
            double z = k * hz;
            for (int j = 1; j < ny-1; ++j) {
                double y = j * hy;
                for (int i = 1; i < nx-1; ++i) {
                    double x = i * hx;
                    const int index = idx(i, j, k);
                    b[index] = sourceFunc(x, y, z);
                }
            }
        }
        
        // Apply boundary conditions
        applyDirichletBC(boundaryFunc);
        
        // Solve the system using SOR
        SOR<TNum, SparseMatrixCSC<TNum>, VectorObj<TNum>> solver(A, b, maxIter, omega);
        solver.solve(u);
    }
    
    // Get the solution at a specific grid point
    TNum getSolution(int i, int j, int k) const {
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) {
            throw std::out_of_range("Grid indices out of range");
        }
        return u[idx(i, j, k)];
    }
    
    // Get the entire solution vector
    const VectorObj<TNum>& getSolution() const {
        return u;
    }
    
    // Get grid dimensions
    int getNx() const { return nx; }
    int getNy() const { return ny; }
    int getNz() const { return nz; }
    
    // Get grid spacing
    double getHx() const { return hx; }
    double getHy() const { return hy; }
    double getHz() const { return hz; }
    
    // Get the physical coordinates for a grid point
    void getCoordinates(int i, int j, int k, double& x, double& y, double& z) const {
        x = i * hx;
        y = j * hy;
        z = k * hz;
    }
};

} // namespace FDM

#endif // POISSON_HPP
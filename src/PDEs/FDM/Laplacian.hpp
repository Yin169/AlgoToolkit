#ifndef LAPLACIAN_HPP
#define LAPLACIAN_HPP

#include "../../Obj/SparseObj.hpp"
#include "../../Obj/VectorObj.hpp"
#include "../../LinearAlgebra/Krylov/ConjugateGradient.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fstream>

enum class BoundaryType { Dirichlet, Neumann, Periodic };

template <typename TNum>
class Poisson3DSolver {
private:
    int nx, ny, nz;           // Number of grid points in each dimension
    double hx, hy, hz;        // Grid spacing in each dimension
    double lx, ly, lz;        // Domain size in each dimension
    SparseMatrixCSC<TNum> A;  // Coefficient matrix for the Laplacian
    VectorObj<TNum> b;        // Right-hand side vector
    VectorObj<TNum> u;        // Solution vector
    
    // Boundary conditions
    BoundaryType bcTypeX, bcTypeY, bcTypeZ;
    
public:
    Poisson3DSolver(int nx, int ny, int nz, 
                   double lx, double ly, double lz,
                   BoundaryType bcTypeX = BoundaryType::Dirichlet,
                   BoundaryType bcTypeY = BoundaryType::Dirichlet,
                   BoundaryType bcTypeZ = BoundaryType::Dirichlet) 
        : nx(nx), ny(ny), nz(nz), 
          lx(lx), ly(ly), lz(lz),
          hx(lx/(nx-1)), hy(ly/(ny-1)), hz(lz/(nz-1)),
          bcTypeX(bcTypeX), bcTypeY(bcTypeY), bcTypeZ(bcTypeZ) {
        
        // Initialize solution vector
        int totalPoints = nx * ny * nz;
        u = VectorObj<TNum>(totalPoints, 0.0);
        b = VectorObj<TNum>(totalPoints, 0.0);
        
        // Build the Laplacian matrix
        buildLaplacianMatrix();
    }
    
    void buildLaplacianMatrix() {
        int totalPoints = nx * ny * nz;
        A = SparseMatrixCSC<TNum>(totalPoints, totalPoints);
        
        // Coefficients for the 7-point stencil
        double cx = 1.0 / (hx * hx);
        double cy = 1.0 / (hy * hy);
        double cz = 1.0 / (hz * hz);
        double cc = -2.0 * (cx + cy + cz); // Center coefficient
        
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    int idx = index(i, j, k);
                    
                    // Handle boundary conditions
                    bool isBoundary = (i == 0 || i == nx-1 || 
                                      j == 0 || j == ny-1 || 
                                      k == 0 || k == nz-1);
                    
                    if (isBoundary && (bcTypeX == BoundaryType::Dirichlet || 
                                      bcTypeY == BoundaryType::Dirichlet || 
                                      bcTypeZ == BoundaryType::Dirichlet)) {
                        // Dirichlet boundary: u = g
                        A.addValue(idx, idx, 1.0);
                    } else {
                        // Interior points: apply the 7-point stencil
                        A.addValue(idx, idx, cc);
                        
                        // X-direction neighbors
                        if (i > 0) A.addValue(idx, index(i-1, j, k), cx);
                        if (i < nx-1) A.addValue(idx, index(i+1, j, k), cx);
                        
                        // Y-direction neighbors
                        if (j > 0) A.addValue(idx, index(i, j-1, k), cy);
                        if (j < ny-1) A.addValue(idx, index(i, j+1, k), cy);
                        
                        // Z-direction neighbors
                        if (k > 0) A.addValue(idx, index(i, j, k-1), cz);
                        if (k < nz-1) A.addValue(idx, index(i, j, k+1), cz);
                    }
                }
            }
        }
        
        A.finalize();
    }
    
    // Helper function to convert 3D indices to 1D index
    inline int index(int i, int j, int k) const {
        return i + j * nx + k * nx * ny;
    }
    
    // Set the right-hand side function f
    void setRHS(const std::function<TNum(double, double, double)>& f) {
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    double x = i * hx;
                    double y = j * hy;
                    double z = k * hz;
                    int idx = index(i, j, k);
                    
                    // For interior points, set the RHS to f(x,y,z)
                    bool isBoundary = (i == 0 || i == nx-1 || 
                                      j == 0 || j == ny-1 || 
                                      k == 0 || k == nz-1);
                    
                    if (isBoundary && (bcTypeX == BoundaryType::Dirichlet || 
                                      bcTypeY == BoundaryType::Dirichlet || 
                                      bcTypeZ == BoundaryType::Dirichlet)) {
                        // For Dirichlet boundaries, set b to the boundary value
                        b[idx] = f(x, y, z);
                    } else {
                        // For interior points, set b to f(x,y,z)
                        b[idx] = f(x, y, z);
                    }
                }
            }
        }
    }
    
    // Set Dirichlet boundary conditions
    void setDirichletBC(const std::function<TNum(double, double, double)>& g) {
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    bool isBoundary = (i == 0 || i == nx-1 || 
                                      j == 0 || j == ny-1 || 
                                      k == 0 || k == nz-1);
                    
                    if (isBoundary) {
                        double x = i * hx;
                        double y = j * hy;
                        double z = k * hz;
                        int idx = index(i, j, k);
                        
                        b[idx] = g(x, y, z);
                    }
                }
            }
        }
    }
    
    // Solve the Poisson equation using Conjugate Gradient
    void solve(int maxIter = 1000, double tol = 1e-6) {
        ConjugateGrad<TNum, SparseMatrixCSC<TNum>, VectorObj<TNum>> cg(A, b, maxIter, tol);
        cg.solve(u);
    }
    
    // Get the solution at a specific grid point
    TNum getSolution(int i, int j, int k) const {
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) {
            throw std::out_of_range("Grid indices out of range");
        }
        return u[index(i, j, k)];
    }

    // Get the entire solution vector
    const VectorObj<TNum>& getSolution() const {
        return u;
    }
    
    // Get grid information
    double getGridSpacingX() const { return hx; }
    double getGridSpacingY() const { return hy; }
    double getGridSpacingZ() const { return hz; }
    int getGridSizeX() const { return nx; }
    int getGridSizeY() const { return ny; }
    int getGridSizeZ() const { return nz; }
};

#endif // LAPLACIAN_HPP
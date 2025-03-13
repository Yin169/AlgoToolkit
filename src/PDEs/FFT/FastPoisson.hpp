#ifndef FAST_POISSON_HPP
#define FAST_POISSON_HPP

#include <complex>
#include <vector>
#include <fftw3.h>
#include <memory>
#include <cmath>
#include <functional>
#include "../../Obj/VectorObj.hpp"
#include "../../Obj/DenseObj.hpp"


enum class BoundaryType { Dirichlet, Neumann, Periodic };

template <typename T>
class FastPoisson3D {
private:
    int nx, ny, nz;
    double dx, dy, dz;
    double Lx, Ly, Lz;

    VectorObj<T> b;
    
    // FFTW plans and arrays
    fftw_complex *in, *out;
    fftw_plan forward_plan, backward_plan;
    
    // Eigenvalues of the Laplacian operator in Fourier space
    std::vector<double> eigenvalues;

    // Boundary conditions
    BoundaryType bcTypeX, bcTypeY, bcTypeZ;
    
    void initializeFFTW() {
        // Allocate memory for FFTW
        in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
        out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
        
        // Create FFTW plans
        forward_plan = fftw_plan_dft_3d(nx, ny, nz, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        backward_plan = fftw_plan_dft_3d(nx, ny, nz, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
        
        // Precompute eigenvalues of the Laplacian in Fourier space
        eigenvalues.resize(nx * ny * nz);
        
        for (int i = 0; i < nx; i++) {
            // Use more accurate wavenumber calculation
            double kx = (i <= nx/2) ? 
                (2.0 * M_PI * static_cast<double>(i)) / Lx : 
                (2.0 * M_PI * static_cast<double>(i - nx)) / Lx;
            
            for (int j = 0; j < ny; j++) {
                double ky = (j <= ny/2) ? 
                    (2.0 * M_PI * static_cast<double>(j)) / Ly : 
                    (2.0 * M_PI * static_cast<double>(j - ny)) / Ly;
                
                for (int k = 0; k < nz; k++) {
                    double kz = (k <= nz/2) ? 
                        (2.0 * M_PI * static_cast<double>(k)) / Lz : 
                        (2.0 * M_PI * static_cast<double>(k - nz)) / Lz;
                    
                    // Use more stable computation of eigenvalues
                    int idx = i * ny * nz + j * nz + k;
                    double kx2 = kx * kx;
                    double ky2 = ky * ky;
                    double kz2 = kz * kz;
                    eigenvalues[idx] = -(kx2 + ky2 + kz2);
                }
            }
        }
    }
    
public:
    FastPoisson3D(int nx_, int ny_, int nz_, double Lx_, double Ly_, double Lz_,
        BoundaryType bcTypeX = BoundaryType::Dirichlet,
        BoundaryType bcTypeY = BoundaryType::Dirichlet,
        BoundaryType bcTypeZ = BoundaryType::Dirichlet) 
        : nx(nx_), ny(ny_), nz(nz_), Lx(Lx_), Ly(Ly_), Lz(Lz_),
        bcTypeX(bcTypeX), 
        bcTypeY(bcTypeY), 
        bcTypeZ(bcTypeZ) {
        dx = Lx / nx;
        dy = Ly / ny;
        dz = Lz / nz;
        
        b = VectorObj<T>(nx * ny * nz);
        initializeFFTW();
    }
    
    ~FastPoisson3D() {
        fftw_destroy_plan(forward_plan);
        fftw_destroy_plan(backward_plan);
        fftw_free(in);
        fftw_free(out);
    }
    
    // Solve the Poisson equation: ∇²u = f
    VectorObj<T> solve() {
        if (b.size() != nx * ny * nz) {
            throw std::invalid_argument("Input vector size does not match grid dimensions");
        }
        
        // Copy f to the FFTW input array
        for (int i = 0; i < nx * ny * nz; i++) {
            in[i][0] = b[i];
            in[i][1] = 0.0;  // Imaginary part is zero
        }
        
        // Forward FFT
        fftw_execute(forward_plan);
        
        // Solve in Fourier space
        for (int i = 0; i < nx * ny * nz; i++) {
            // Skip DC component (i=0,j=0,k=0) to handle constant offset
            if (std::abs(eigenvalues[i]) < 1e-10) {
                out[i][0] = 0.0;
                out[i][1] = 0.0;
            } else {
                // Divide by eigenvalue
                out[i][0] /= eigenvalues[i];
                out[i][1] /= eigenvalues[i];
            }
        }
        
        // Backward FFT
        fftw_execute(backward_plan);
        
        // Copy result to output vector and normalize
        VectorObj<T> u(nx * ny * nz);
        // FFTW normalization factor for 3D transform
        const double normalization = 1.0 / static_cast<double>(nx * ny * nz);
        
        for (int i = 0; i < nx * ny * nz; i++) {
            // Use careful type conversion for numerical stability
            u[i] = static_cast<T>(in[i][0] * normalization);
        }
        
        return u;
    }

    
    // Set the right-hand side function f
    void setRHS(const std::function<T(double, double, double)>& f) {
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    double x = i * dx;
                    double y = j * dy;
                    double z = k * dz;
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
    void setDirichletBC(const std::function<T(double, double, double)>& g) {
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    bool isBoundary = (i == 0 || i == nx-1 || 
                                      j == 0 || j == ny-1 || 
                                      k == 0 || k == nz-1);
                    
                    if (isBoundary) {
                        double x = i * dx;
                        double y = j * dy;
                        double z = k * dz;
                        int idx = index(i, j, k);
                        
                        b[idx] = g(x, y, z);
                    }
                }
            }
        }
    }
    
    
    // Get grid dimensions
    int getNx() const { return nx; }
    int getNy() const { return ny; }
    int getNz() const { return nz; }
    
    // Get grid spacing
    double getDx() const { return dx; }
    double getDy() const { return dy; }
    double getDz() const { return dz; }
    
    // Get domain lengths
    double getLx() const { return Lx; }
    double getLy() const { return Ly; }
    double getLz() const { return Lz; }
    
    // Convert 3D indices to 1D index
    int index(int i, int j, int k) const {
        return i * ny * nz + j * nz + k;
    }
    
    // Get solution value at specific grid point
    T getValue(const VectorObj<T>& solution, int i, int j, int k) const {
        return solution[index(i, j, k)];
    }
};

#endif // FAST_POISSON_HPP

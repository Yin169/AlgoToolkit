#ifndef FAST_POISSON_HPP
#define FAST_POISSON_HPP

#include <complex>
#include <vector>
#include <fftw3.h>
#include <memory>
#include <cmath>
#include "../../Obj/VectorObj.hpp"
#include "../../Obj/DenseObj.hpp"

template <typename T>
class FastPoisson3D {
private:
    int nx, ny, nz;
    double dx, dy, dz;
    double Lx, Ly, Lz;
    
    // FFTW plans and arrays
    fftw_complex *in, *out;
    fftw_plan forward_plan, backward_plan;
    
    // Eigenvalues of the Laplacian operator in Fourier space
    std::vector<double> eigenvalues;
    
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
            double kx = (i <= nx/2) ? 2 * M_PI * i / Lx : 2 * M_PI * (i - nx) / Lx;
            
            for (int j = 0; j < ny; j++) {
                double ky = (j <= ny/2) ? 2 * M_PI * j / Ly : 2 * M_PI * (j - ny) / Ly;
                
                for (int k = 0; k < nz; k++) {
                    double kz = (k <= nz/2) ? 2 * M_PI * k / Lz : 2 * M_PI * (k - nz) / Lz;
                    
                    // Eigenvalue of the Laplacian: -(kx^2 + ky^2 + kz^2)
                    int idx = i * ny * nz + j * nz + k;
                    eigenvalues[idx] = -(kx*kx + ky*ky + kz*kz);
                }
            }
        }
    }
    
public:
    FastPoisson3D(int nx_, int ny_, int nz_, double Lx_, double Ly_, double Lz_) 
        : nx(nx_), ny(ny_), nz(nz_), Lx(Lx_), Ly(Ly_), Lz(Lz_) {
        dx = Lx / nx;
        dy = Ly / ny;
        dz = Lz / nz;
        
        initializeFFTW();
    }
    
    ~FastPoisson3D() {
        fftw_destroy_plan(forward_plan);
        fftw_destroy_plan(backward_plan);
        fftw_free(in);
        fftw_free(out);
    }
    
    // Solve the Poisson equation: ∇²u = f
    VectorObj<T> solve(const VectorObj<T>& f) {
        if (f.size() != nx * ny * nz) {
            throw std::invalid_argument("Input vector size does not match grid dimensions");
        }
        
        // Copy f to the FFTW input array
        for (int i = 0; i < nx * ny * nz; i++) {
            in[i][0] = f[i];
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
        double normalization = 1.0 / (nx * ny * nz);
        
        for (int i = 0; i < nx * ny * nz; i++) {
            u[i] = in[i][0] * normalization;  // Take real part and normalize
        }
        
        return u;
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
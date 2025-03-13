#ifndef SPECTRAL_METHOD_HPP
#define SPECTRAL_METHOD_HPP

#include "../../Obj/VectorObj.hpp"
#include "../../Obj/DenseObj.hpp"
#include "../../ODE/RungeKutta.hpp"
#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <functional>
#include <algorithm>
#include <string>
#include <memory>
#include <fftw3.h>
#include <random>

enum class SpectralTimeIntegration {
    EULER,
    RK4,
    AB3  // Adams-Bashforth 3rd order
};

template <typename TNum>
class SpectralNavierStokesSolver {
private:
    // Grid dimensions
    int nx, ny, nz;
    TNum Lx, Ly, Lz;  // Domain sizes
    TNum dt;
    TNum Re;  // Reynolds number
    TNum nu;  // Kinematic viscosity (1/Re)
    
    // Wavenumbers
    std::vector<TNum> kx, ky, kz;
    std::vector<std::vector<std::vector<TNum>>> k_squared;
    
    // Velocity components in physical space
    VectorObj<TNum> u, v, w;
    
    // Velocity components in spectral space (complex)
   VectorObj<std::complex<TNum>> u_hat, v_hat, w_hat;
   VectorObj<std::complex<TNum>> u_hat_new, v_hat_new, w_hat_new;
    
    // Nonlinear terms
   VectorObj<std::complex<TNum>> N_u_hat, N_v_hat, N_w_hat;
   VectorObj<std::complex<TNum>> N_u_hat_old, N_v_hat_old, N_w_hat_old;
   VectorObj<std::complex<TNum>> N_u_hat_older, N_v_hat_older, N_w_hat_older;
    
    // Pressure in spectral space
   VectorObj<std::complex<TNum>> p_hat;
    
    // FFTW plans
    fftw_plan forward_plan_u, backward_plan_u;
    fftw_plan forward_plan_v, backward_plan_v;
    fftw_plan forward_plan_w, backward_plan_w;
    
    // Temporary arrays for FFT
    fftw_complex *u_complex, *v_complex, *w_complex;
    fftw_complex *u_complex_out, *v_complex_out, *w_complex_out;
    
    // Dealias mask (2/3 rule)
    std::vector<TNum> dealias_mask;
    
    // Time integration method
    SpectralTimeIntegration time_method = SpectralTimeIntegration::RK4;
    std::unique_ptr<RungeKutta<TNum,VectorObj<std::complex<TNum>>>> rk_solver;
    
    // External forcing
   VectorObj<std::complex<TNum>> f_u_hat, f_v_hat, f_w_hat;
    bool has_external_forces = false;
    
    // Helper methods
    int idx(int i, int j, int k) const {
        return i + j * nx + k * nx * ny;
    }
    
    void initializeWavenumbers() {
        // Initialize wavenumbers
        kx.resize(nx);
        ky.resize(ny);
        kz.resize(nz);
        
        // Set up wavenumbers for FFTW
        for (int i = 0; i <= nx/2; i++) {
            kx[i] = 2.0 * M_PI * i / Lx;
        }
        for (int i = nx/2 + 1; i < nx; i++) {
            kx[i] = 2.0 * M_PI * (i - nx) / Lx;
        }
        
        for (int j = 0; j <= ny/2; j++) {
            ky[j] = 2.0 * M_PI * j / Ly;
        }
        for (int j = ny/2 + 1; j < ny; j++) {
            ky[j] = 2.0 * M_PI * (j - ny) / Ly;
        }
        
        for (int k = 0; k <= nz/2; k++) {
            kz[k] = 2.0 * M_PI * k / Lz;
        }
        for (int k = nz/2 + 1; k < nz; k++) {
            kz[k] = 2.0 * M_PI * (k - nz) / Lz;
        }
        
        // Precompute k^2 for efficiency
        k_squared.resize(nx, std::vector<std::vector<TNum>>(ny, std::vector<TNum>(nz)));
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    k_squared[i][j][k] = kx[i]*kx[i] + ky[j]*ky[j] + kz[k]*kz[k];
                }
            }
        }
        
        // Initialize dealiasing mask (2/3 rule)
        dealias_mask.resize(nx * ny * nz, 1.0);
        int kx_max = nx / 3;
        int ky_max = ny / 3;
        int kz_max = nz / 3;
        
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    if (std::abs(i <= nx/2 ? i : i - nx) > kx_max ||
                        std::abs(j <= ny/2 ? j : j - ny) > ky_max ||
                        std::abs(k <= nz/2 ? k : k - nz) > kz_max) {
                        dealias_mask[idx(i, j, k)] = 0.0;
                    }
                }
            }
        }
    }
    
    void initializeFFTW() {
        // Allocate memory for FFTW
        u_complex = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
        v_complex = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
        w_complex = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
        
        u_complex_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
        v_complex_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
        w_complex_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
        
        // Create FFTW plans
        forward_plan_u = fftw_plan_dft_3d(nx, ny, nz, u_complex, u_complex_out, FFTW_FORWARD, FFTW_MEASURE);
        backward_plan_u = fftw_plan_dft_3d(nx, ny, nz, u_complex, u_complex_out, FFTW_BACKWARD, FFTW_MEASURE);
        
        forward_plan_v = fftw_plan_dft_3d(nx, ny, nz, v_complex, v_complex_out, FFTW_FORWARD, FFTW_MEASURE);
        backward_plan_v = fftw_plan_dft_3d(nx, ny, nz, v_complex, v_complex_out, FFTW_BACKWARD, FFTW_MEASURE);
        
        forward_plan_w = fftw_plan_dft_3d(nx, ny, nz, w_complex, w_complex_out, FFTW_FORWARD, FFTW_MEASURE);
        backward_plan_w = fftw_plan_dft_3d(nx, ny, nz, w_complex, w_complex_out, FFTW_BACKWARD, FFTW_MEASURE);
    }
    
    void cleanupFFTW() {
        // Destroy FFTW plans
        fftw_destroy_plan(forward_plan_u);
        fftw_destroy_plan(backward_plan_u);
        fftw_destroy_plan(forward_plan_v);
        fftw_destroy_plan(backward_plan_v);
        fftw_destroy_plan(forward_plan_w);
        fftw_destroy_plan(backward_plan_w);
        
        // Free FFTW memory
        fftw_free(u_complex);
        fftw_free(v_complex);
        fftw_free(w_complex);
        fftw_free(u_complex_out);
        fftw_free(v_complex_out);
        fftw_free(w_complex_out);
    }
    
    void forwardTransform() {
        // Copy physical space data to FFTW arrays
        for (int i = 0; i < nx * ny * nz; i++) {
            u_complex[i][0] = u[i];
            u_complex[i][1] = 0.0;
            
            v_complex[i][0] = v[i];
            v_complex[i][1] = 0.0;
            
            w_complex[i][0] = w[i];
            w_complex[i][1] = 0.0;
        }
        
        // Execute forward FFT
        fftw_execute(forward_plan_u);
        fftw_execute(forward_plan_v);
        fftw_execute(forward_plan_w);
        
        // Copy results to spectral arrays
        TNum normalization = 1.0 / (nx * ny * nz);
        for (int i = 0; i < nx * ny * nz; i++) {
            u_hat[i] = std::complex<TNum>(u_complex_out[i][0], u_complex_out[i][1]) * normalization;
            v_hat[i] = std::complex<TNum>(v_complex_out[i][0], v_complex_out[i][1]) * normalization;
            w_hat[i] = std::complex<TNum>(w_complex_out[i][0], w_complex_out[i][1]) * normalization;
        }
    }
    
    void backwardTransform() {
        // Copy spectral data to FFTW arrays
        for (int i = 0; i < nx * ny * nz; i++) {
            u_complex[i][0] = u_hat[i].real();
            u_complex[i][1] = u_hat[i].imag();
            
            v_complex[i][0] = v_hat[i].real();
            v_complex[i][1] = v_hat[i].imag();
            
            w_complex[i][0] = w_hat[i].real();
            w_complex[i][1] = w_hat[i].imag();
        }
        
        // Execute backward FFT
        fftw_execute(backward_plan_u);
        fftw_execute(backward_plan_v);
        fftw_execute(backward_plan_w);
        
        // Copy results to physical arrays
        for (int i = 0; i < nx * ny * nz; i++) {
            u[i] = u_complex_out[i][0];
            v[i] = v_complex_out[i][0];
            w[i] = w_complex_out[i][0];
        }
    }
    
    void computeNonlinearTerms() {
        // Compute nonlinear terms in physical space
        VectorObj<TNum> uu(nx * ny * nz), uv(nx * ny * nz), uw(nx * ny * nz);
        VectorObj<TNum> vu(nx * ny * nz), vv(nx * ny * nz), vw(nx * ny * nz);
        VectorObj<TNum> wu(nx * ny * nz), wv(nx * ny * nz), ww(nx * ny * nz);
        
        // Compute products in physical space
        for (int i = 0; i < nx * ny * nz; i++) {
            uu[i] = u[i] * u[i];
            uv[i] = u[i] * v[i];
            uw[i] = u[i] * w[i];
            
            vu[i] = v[i] * u[i];
            vv[i] = v[i] * v[i];
            vw[i] = v[i] * w[i];
            
            wu[i] = w[i] * u[i];
            wv[i] = w[i] * v[i];
            ww[i] = w[i] * w[i];
        }
        
        // Transform products to spectral space
       VectorObj<std::complex<TNum>> uu_hat(nx * ny * nz), uv_hat(nx * ny * nz), uw_hat(nx * ny * nz);
       VectorObj<std::complex<TNum>> vu_hat(nx * ny * nz), vv_hat(nx * ny * nz), vw_hat(nx * ny * nz);
       VectorObj<std::complex<TNum>> wu_hat(nx * ny * nz), wv_hat(nx * ny * nz), ww_hat(nx * ny * nz);
        
        // Transform each product
        transformField(uu, uu_hat);
        transformField(uv, uv_hat);
        transformField(uw, uw_hat);
        transformField(vu, vu_hat);
        transformField(vv, vv_hat);
        transformField(vw, vw_hat);
        transformField(wu, wu_hat);
        transformField(wv, wv_hat);
        transformField(ww, ww_hat);
        
        // Compute derivatives in spectral space and form nonlinear terms
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    int index = idx(i, j, k);
                    std::complex<TNum> I(0, 1);
                    
                    // Apply dealiasing
                    TNum mask = dealias_mask[index];
                    
                    // N_u = -u∂u/∂x - v∂u/∂y - w∂u/∂z
                    N_u_hat[index] = -I * (
                        kx[i] * uu_hat[index] +
                        ky[j] * vu_hat[index] +
                        kz[k] * wu_hat[index]
                    ) * mask;
                    
                    // N_v = -u∂v/∂x - v∂v/∂y - w∂v/∂z
                    N_v_hat[index] = -I * (
                        kx[i] * uv_hat[index] +
                        ky[j] * vv_hat[index] +
                        kz[k] * wv_hat[index]
                    ) * mask;
                    
                    // N_w = -u∂w/∂x - v∂w/∂y - w∂w/∂z
                    N_w_hat[index] = -I * (
                        kx[i] * uw_hat[index] +
                        ky[j] * vw_hat[index] +
                        kz[k] * ww_hat[index]
                    ) * mask;
                }
            }
        }
    }
    
    void transformField(const VectorObj<TNum>& field,VectorObj<std::complex<TNum>>& field_hat) {
        // Temporary FFTW arrays
        fftw_complex *temp_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
        fftw_complex *temp_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
        
        // Create temporary plan
        fftw_plan temp_plan = fftw_plan_dft_3d(nx, ny, nz, temp_in, temp_out, FFTW_FORWARD, FFTW_ESTIMATE);
        
        // Copy field to FFTW array
        for (int i = 0; i < nx * ny * nz; i++) {
            temp_in[i][0] = field[i];
            temp_in[i][1] = 0.0;
        }
        
        // Execute FFT
        fftw_execute(temp_plan);
        
        // Copy result to field_hat
        TNum normalization = 1.0 / (nx * ny * nz);
        for (int i = 0; i < nx * ny * nz; i++) {
            field_hat[i] = std::complex<TNum>(temp_out[i][0], temp_out[i][1]) * normalization;
        }
        
        // Clean up
        fftw_destroy_plan(temp_plan);
        fftw_free(temp_in);
        fftw_free(temp_out);
    }
    
    void computePressure() {
        // Compute pressure in spectral space using the incompressibility constraint
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    int index = idx(i, j, k);
                    
                    // Skip k = (0,0,0) mode (mean pressure is arbitrary)
                    if (i == 0 && j == 0 && k == 0) {
                        p_hat[index] = 0.0;
                        continue;
                    }
                    
                    std::complex<TNum> I(0, 1);
                    TNum k2 = k_squared[i][j][k];
                    
                    // Compute pressure from incompressibility: ∇·u = 0
                    // This leads to: ∇²p = -∇·(u·∇)u
                    p_hat[index] = -I * (
                        kx[i] * N_u_hat[index] +
                        ky[j] * N_v_hat[index] +
                        kz[k] * N_w_hat[index]
                    ) / k2;
                }
            }
        }
    }
    
    void applyProjection() {
        // Project velocity field to ensure incompressibility (remove any divergence)
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    int index = idx(i, j, k);
                    
                    // Skip k = (0,0,0) mode
                    if (i == 0 && j == 0 && k == 0) {
                        continue;
                    }
                    
                    std::complex<TNum> I(0, 1);
                    TNum k2 = k_squared[i][j][k];
                    
                    // Compute projection tensor: δ_ij - k_i k_j / k²
                    std::complex<TNum> k_dot_u = kx[i] * u_hat[index] + 
                                                ky[j] * v_hat[index] + 
                                                kz[k] * w_hat[index];
                    
                    // Apply projection to ensure incompressibility
                    u_hat[index] -= (kx[i] * k_dot_u) / k2;
                    v_hat[index] -= (ky[j] * k_dot_u) / k2;
                    w_hat[index] -= (kz[k] * k_dot_u) / k2;
                }
            }
        }
    }
    
    // Time integration methods
    void integrateEuler() {
        // Store old nonlinear terms for multi-step methods
        N_u_hat_older = N_u_hat_old;
        N_v_hat_older = N_v_hat_old;
        N_w_hat_older = N_w_hat_old;
        
        N_u_hat_old = N_u_hat;
        N_v_hat_old = N_v_hat;
        N_w_hat_old = N_w_hat;
        
        // Compute nonlinear terms
        computeNonlinearTerms();
        
        // Integrate using Euler method
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    int index = idx(i, j, k);
                    
                    // Viscous term: ν∇²u = -νk²u in spectral space
                    TNum k2 = k_squared[i][j][k];
                    std::complex<TNum> visc_term_u = -nu * k2 * u_hat[index];
                    std::complex<TNum> visc_term_v = -nu * k2 * v_hat[index];
                    std::complex<TNum> visc_term_w = -nu * k2 * w_hat[index];
                    
                    // External forcing if present
                    std::complex<TNum> force_u = has_external_forces ? f_u_hat[index] : std::complex<TNum>(0, 0);
                    std::complex<TNum> force_v = has_external_forces ? f_v_hat[index] : std::complex<TNum>(0, 0);
                    std::complex<TNum> force_w = has_external_forces ? f_w_hat[index] : std::complex<TNum>(0, 0);
                    
                    // Update velocity in spectral space
                    u_hat[index] += dt * (N_u_hat[index] + visc_term_u + force_u);
                    v_hat[index] += dt * (N_v_hat[index] + visc_term_v + force_v);
                    w_hat[index] += dt * (N_w_hat[index] + visc_term_w + force_w);
                }
            }
        }
        
        // Project to ensure incompressibility
        applyProjection();
    }
    
    void integrateAB3() {
        // Adams-Bashforth 3rd order method
        // Requires two previous evaluations of nonlinear terms
        
        // Store old nonlinear terms
        N_u_hat_older = N_u_hat_old;
        N_v_hat_older = N_v_hat_old;
        N_w_hat_older = N_w_hat_old;
        
        N_u_hat_old = N_u_hat;
        N_v_hat_old = N_v_hat;
        N_w_hat_old = N_w_hat;
        
        // Compute new nonlinear terms
        computeNonlinearTerms();
        
        // AB3 coefficients
        TNum ab3_coef1 = 23.0 / 12.0;
        TNum ab3_coef2 = -16.0 / 12.0;
        TNum ab3_coef3 = 5.0 / 12.0;
        
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    int index = idx(i, j, k);
                    
                    // Viscous term (implicit)
                    TNum k2 = k_squared[i][j][k];
                    std::complex<TNum> exp_factor = std::exp(-nu * k2 * dt);
                    
                    // External forcing if present
                    std::complex<TNum> force_u = has_external_forces ? f_u_hat[index] : std::complex<TNum>(0, 0);
                    std::complex<TNum> force_v = has_external_forces ? f_v_hat[index] : std::complex<TNum>(0, 0);
                    std::complex<TNum> force_w = has_external_forces ? f_w_hat[index] : std::complex<TNum>(0, 0);
                    
                    // AB3 for nonlinear terms
                    std::complex<TNum> N_u_ab3 = ab3_coef1 * N_u_hat[index] + 
                                                ab3_coef2 * N_u_hat_old[index] + 
                                                ab3_coef3 * N_u_hat_older[index];
                    
                    std::complex<TNum> N_v_ab3 = ab3_coef1 * N_v_hat[index] + 
                                                ab3_coef2 * N_v_hat_old[index] + 
                                                ab3_coef3 * N_v_hat_older[index];
                    
                    std::complex<TNum> N_w_ab3 = ab3_coef1 * N_w_hat[index] + 
                                                ab3_coef2 * N_w_hat_old[index] + 
                                                ab3_coef3 * N_w_hat_older[index];
                    
                    // Update velocity (semi-implicit: explicit for nonlinear, implicit for viscous)
                    u_hat[index] = exp_factor * u_hat[index] + dt * (N_u_ab3 + force_u);
                    v_hat[index] = exp_factor * v_hat[index] + dt * (N_v_ab3 + force_v);
                    w_hat[index] = exp_factor * w_hat[index] + dt * (N_w_ab3 + force_w);
                }
            }
        }
        
        // Project to ensure incompressibility
        applyProjection();
    }
    
    void integrateRK4() {
        // RK4 integration using the RungeKutta class
        auto u_system = [this](const VectorObj<std::complex<TNum>>& u_vec) ->VectorObj<std::complex<TNum>> {
            // Store current state
           VectorObj<std::complex<TNum>> u_temp = u_hat;
           VectorObj<std::complex<TNum>> v_temp = v_hat;
           VectorObj<std::complex<TNum>> w_temp = w_hat;
            
            // Set current state to input
            u_hat = u_vec;
            
            // Transform to physical space
            backwardTransform();
            
            // Compute nonlinear terms
            computeNonlinearTerms();
            
            // Compute right-hand side
           VectorObj<std::complex<TNum>> result(nx * ny * nz);
            for (int i = 0; i < nx; i++) {
                for (int j = 0; j < ny; j++) {
                    for (int k = 0; k < nz; k++) {
                        int index = idx(i, j, k);
                        
                        // Viscous term
                        TNum k2 = k_squared[i][j][k];
                        std::complex<TNum> visc_term = -nu * k2 * u_hat[index];
                        
                        // External forcing
                        std::complex<TNum> force = has_external_forces ? f_u_hat[index] : std::complex<TNum>(0, 0);
                        
                        // RHS = nonlinear + viscous + forcing
                        result[index] = N_u_hat[index] + visc_term + force;
                    }
                }
            }
            
            // Restore original state
            u_hat = u_temp;
            v_hat = v_temp;
            w_hat = w_temp;
            
            return result;
        };
        
        auto v_system = [this](const VectorObj<std::complex<TNum>>& v_vec) ->VectorObj<std::complex<TNum>> {
            // Similar to u_system but for v component
           VectorObj<std::complex<TNum>> u_temp = u_hat;
           VectorObj<std::complex<TNum>> v_temp = v_hat;
           VectorObj<std::complex<TNum>> w_temp = w_hat;
            
            v_hat = v_vec;
            
            backwardTransform();
            computeNonlinearTerms();
            
           VectorObj<std::complex<TNum>> result(nx * ny * nz);
            for (int index = 0; index < nx * ny * nz; index++) {
                int i = index % nx;
                int j = (index / nx) % ny;
                int k = index / (nx * ny);
                
                TNum k2 = k_squared[i][j][k];
                std::complex<TNum> visc_term = -nu * k2 * v_hat[index];
                std::complex<TNum> force = has_external_forces ? f_v_hat[index] : std::complex<TNum>(0, 0);
                
                result[index] = N_v_hat[index] + visc_term + force;
            }
            
            u_hat = u_temp;
            v_hat = v_temp;
            w_hat = w_temp;
            
            return result;
        };
        
        auto w_system = [this](const VectorObj<std::complex<TNum>>& w_vec) ->VectorObj<std::complex<TNum>> {
            // Similar to u_system but for w component
           VectorObj<std::complex<TNum>> u_temp = u_hat;
           VectorObj<std::complex<TNum>> v_temp = v_hat;
           VectorObj<std::complex<TNum>> w_temp = w_hat;
            
            w_hat = w_vec;
            
            backwardTransform();
            computeNonlinearTerms();
            
           VectorObj<std::complex<TNum>> result(nx * ny * nz);
            for (int index = 0; index < nx * ny * nz; index++) {
                int i = index % nx;
                int j = (index / nx) % ny;
                int k = index / (nx * ny);
                
                TNum k2 = k_squared[i][j][k];
                std::complex<TNum> visc_term = -nu * k2 * w_hat[index];
                std::complex<TNum> force = has_external_forces ? f_w_hat[index] : std::complex<TNum>(0, 0);
                
                result[index] = N_w_hat[index] + visc_term + force;
            }
            
            u_hat = u_temp;
            v_hat = v_temp;
            w_hat = w_temp;
            
            return result;
        };
        
        // Solve for each velocity component using RK4
        rk_solver->solve(u_hat, u_system, dt, 1);
        rk_solver->solve(v_hat, v_system, dt, 1);
        rk_solver->solve(w_hat, w_system, dt, 1);
        
        // Project to ensure incompressibility
        applyProjection();
    }
    
    void integrate() {
        // Perform time integration based on selected method
        switch (time_method) {
            case SpectralTimeIntegration::EULER:
                integrateEuler();
                break;
            case SpectralTimeIntegration::RK4:
                integrateRK4();
                break;
            case SpectralTimeIntegration::AB3:
                integrateAB3();
                break;
        }
    }
    
    // Calculate CFL number for adaptive time stepping
    TNum calculateCFL() const {
        // Find maximum velocity in physical space
        TNum max_vel = 0.0;
        for (int i = 0; i < nx * ny * nz; i++) {
            TNum vel_mag = std::sqrt(u[i]*u[i] + v[i]*v[i] + w[i]*w[i]);
            max_vel = std::max(max_vel, vel_mag);
        }
        
        // Calculate minimum grid spacing
        TNum dx = Lx / nx;
        TNum dy = Ly / ny;
        TNum dz = Lz / nz;
        TNum min_dx = std::min(std::min(dx, dy), dz);
        
        // CFL = max_vel * dt / min_dx
        return max_vel * dt / min_dx;
    }

    public:
    // Calculate kinetic energy
    TNum calculateKineticEnergy() const {
        TNum energy = 0.0;
        
        // Sum up energy in physical space
        for (int i = 0; i < nx * ny * nz; i++) {
            energy += 0.5 * (u[i]*u[i] + v[i]*v[i] + w[i]*w[i]);
        }
        
        // Normalize by domain volume
        return energy / (nx * ny * nz);
    }
    
    // Calculate enstrophy (integral of squared vorticity)
    TNum calculateEnstrophy() const {
        // Calculate vorticity in spectral space
       VectorObj<std::complex<TNum>> omega_x_hat(nx * ny * nz);
       VectorObj<std::complex<TNum>> omega_y_hat(nx * ny * nz);
       VectorObj<std::complex<TNum>> omega_z_hat(nx * ny * nz);
        
        std::complex<TNum> I(0, 1);
        
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    int index = idx(i, j, k);
                    
                    // ω_x = ∂w/∂y - ∂v/∂z
                    omega_x_hat[index] = I * (ky[j] * w_hat[index] - kz[k] * v_hat[index]);
                    
                    // ω_y = ∂u/∂z - ∂w/∂x
                    omega_y_hat[index] = I * (kz[k] * u_hat[index] - kx[i] * w_hat[index]);
                    
                    // ω_z = ∂v/∂x - ∂u/∂y
                    omega_z_hat[index] = I * (kx[i] * v_hat[index] - ky[j] * u_hat[index]);
                }
            }
        }
        
        // Transform vorticity to physical space
        VectorObj<TNum> omega_x(nx * ny * nz);
        VectorObj<TNum> omega_y(nx * ny * nz);
        VectorObj<TNum> omega_z(nx * ny * nz);
        
        // Temporary arrays for inverse FFT
        fftw_complex *temp_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
        fftw_complex *temp_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
        fftw_plan temp_plan = fftw_plan_dft_3d(nx, ny, nz, temp_in, temp_out, FFTW_BACKWARD, FFTW_ESTIMATE);
        
        // Transform omega_x
        for (int i = 0; i < nx * ny * nz; i++) {
            temp_in[i][0] = omega_x_hat[i].real();
            temp_in[i][1] = omega_x_hat[i].imag();
        }
        fftw_execute(temp_plan);
        for (int i = 0; i < nx * ny * nz; i++) {
            omega_x[i] = temp_out[i][0];
        }
        
        // Transform omega_y
        for (int i = 0; i < nx * ny * nz; i++) {
            temp_in[i][0] = omega_y_hat[i].real();
            temp_in[i][1] = omega_y_hat[i].imag();
        }
        fftw_execute(temp_plan);
        for (int i = 0; i < nx * ny * nz; i++) {
            omega_y[i] = temp_out[i][0];
        }
        
        // Transform omega_z
        for (int i = 0; i < nx * ny * nz; i++) {
            temp_in[i][0] = omega_z_hat[i].real();
            temp_in[i][1] = omega_z_hat[i].imag();
        }
        fftw_execute(temp_plan);
        for (int i = 0; i < nx * ny * nz; i++) {
            omega_z[i] = temp_out[i][0];
        }
        
        // Clean up
        fftw_destroy_plan(temp_plan);
        fftw_free(temp_in);
        fftw_free(temp_out);
        
        // Calculate enstrophy
        TNum enstrophy = 0.0;
        for (int i = 0; i < nx * ny * nz; i++) {
            enstrophy += 0.5 * (omega_x[i]*omega_x[i] + omega_y[i]*omega_y[i] + omega_z[i]*omega_z[i]);
        }
        
        // Normalize by domain volume
        return enstrophy / (nx * ny * nz);
    }

public:
    // Constructor
    SpectralNavierStokesSolver(int nx_, int ny_, int nz_, 
                              TNum Lx_, TNum Ly_, TNum Lz_,
                              TNum dt_, TNum Re_)
        : nx(nx_), ny(ny_), nz(nz_),
          Lx(Lx_), Ly(Ly_), Lz(Lz_),
          dt(dt_), Re(Re_), nu(1.0/Re_) {
        
        // Initialize arrays
        int n = nx * ny * nz;
        u.resize(n, 0.0);
        v.resize(n, 0.0);
        w.resize(n, 0.0);
        
        u_hat.resize(n, std::complex<TNum>(0, 0));
        v_hat.resize(n, std::complex<TNum>(0, 0));
        w_hat.resize(n, std::complex<TNum>(0, 0));
        
        u_hat_new.resize(n, std::complex<TNum>(0, 0));
        v_hat_new.resize(n, std::complex<TNum>(0, 0));
        w_hat_new.resize(n, std::complex<TNum>(0, 0));
        
        N_u_hat.resize(n, std::complex<TNum>(0, 0));
        N_v_hat.resize(n, std::complex<TNum>(0, 0));
        N_w_hat.resize(n, std::complex<TNum>(0, 0));
        
        N_u_hat_old.resize(n, std::complex<TNum>(0, 0));
        N_v_hat_old.resize(n, std::complex<TNum>(0, 0));
        N_w_hat_old.resize(n, std::complex<TNum>(0, 0));
        
        N_u_hat_older.resize(n, std::complex<TNum>(0, 0));
        N_v_hat_older.resize(n, std::complex<TNum>(0, 0));
        N_w_hat_older.resize(n, std::complex<TNum>(0, 0));
        
        p_hat.resize(n, std::complex<TNum>(0, 0));
        
        f_u_hat.resize(n, std::complex<TNum>(0, 0));
        f_v_hat.resize(n, std::complex<TNum>(0, 0));
        f_w_hat.resize(n, std::complex<TNum>(0, 0));
        
        // Initialize wavenumbers and dealiasing mask
        initializeWavenumbers();
        
        // Initialize FFTW
        initializeFFTW();
        
        // Initialize RK4 solver
        rk_solver = std::make_unique<RungeKutta<TNum,VectorObj<std::complex<TNum>>>>();
    }
    
    // Destructor
    ~SpectralNavierStokesSolver() {
        cleanupFFTW();
    }
    
    // Set initial conditions in physical space
    void setInitialConditions(const VectorObj<TNum>& u0, const VectorObj<TNum>& v0, const VectorObj<TNum>& w0) {
        u = u0;
        v = v0;
        w = w0;
        
        // Transform to spectral space
        forwardTransform();
        
        // Project to ensure initial incompressibility
        applyProjection();
        
        // Transform back to physical space
        backwardTransform();
    }
    
    // Set external forcing in spectral space
    void setExternalForcing(const VectorObj<std::complex<TNum>>& f_u, 
                           const VectorObj<std::complex<TNum>>& f_v,
                           const VectorObj<std::complex<TNum>>& f_w) {
        f_u_hat = f_u;
        f_v_hat = f_v;
        f_w_hat = f_w;
        has_external_forces = true;
    }
    
    // Set time integration method
    void setTimeIntegrationMethod(SpectralTimeIntegration method) {
        time_method = method;
    }
    
    // Set time step
    void setTimeStep(TNum dt_) {
        dt = dt_;
    }
    
    // Advance simulation by one time step
    void step() {
        // Transform to spectral space
        forwardTransform();
        
        // Perform time integration
        integrate();
        
        // Transform back to physical space
        backwardTransform();
    }
    
    // Solve for multiple time steps
    void solve(int num_steps, int output_frequency = 1,
              std::function<void(const VectorObj<TNum>&, const VectorObj<TNum>&, 
                                const VectorObj<TNum>&, TNum, TNum)> output_callback = nullptr) {
        TNum time = 0.0;
        
        for (int tstep = 0; tstep < num_steps; tstep++) {
            // Output if needed
            if (output_callback && tstep % output_frequency == 0) {
                output_callback(u, v, w, time, tstep);
            }
            
            // Advance one time step
            step();
            time += dt;
        }
        
        // Final output
        if (output_callback && num_steps % output_frequency != 0) {
            output_callback(u, v, w, time, num_steps);
        }
    }
    
    // Get velocity fields
    const VectorObj<TNum>& getU() const { return u; }
    const VectorObj<TNum>& getV() const { return v; }
    const VectorObj<TNum>& getW() const { return w; }
    
    // Get grid dimensions
    int getNx() const { return nx; }
    int getNy() const { return ny; }
    int getNz() const { return nz; }
    
    // Get domain sizes
    TNum getLx() const { return Lx; }
    TNum getLy() const { return Ly; }
    TNum getLz() const { return Lz; }
    
    // Get time step
    TNum getDt() const { return dt; }
    
    // Get Reynolds number
    TNum getReynoldsNumber() const { return Re; }
    
    // Calculate divergence (should be close to zero)
    VectorObj<TNum> calculateDivergence() const {
        VectorObj<TNum> div(nx * ny * nz, 0.0);
        
        // Calculate divergence in spectral space
       VectorObj<std::complex<TNum>> div_hat(nx * ny * nz);
        std::complex<TNum> I(0, 1);
        
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    int index = idx(i, j, k);
                    div_hat[index] = I * (kx[i] * u_hat[index] + ky[j] * v_hat[index] + kz[k] * w_hat[index]);
                }
            }
        }
        
        // Transform to physical space
        fftw_complex *temp_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
        fftw_complex *temp_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
		fftw_plan temp_plan = fftw_plan_dft_3d(nx, ny, nz, temp_in, temp_out, FFTW_BACKWARD, FFTW_ESTIMATE);
        
        // Copy divergence to FFTW array
        for (int i = 0; i < nx * ny * nz; i++) {
            temp_in[i][0] = div_hat[i].real();
            temp_in[i][1] = div_hat[i].imag();
        }
        
        // Execute FFT
        fftw_execute(temp_plan);
        
        // Copy result to divergence array
        for (int i = 0; i < nx * ny * nz; i++) {
            div[i] = temp_out[i][0];
        }
        
        // Clean up
        fftw_destroy_plan(temp_plan);
        fftw_free(temp_in);
        fftw_free(temp_out);
        
        return div;
    }
    
    // Get vorticity field
    void getVorticityField(VectorObj<TNum>& omega_x, VectorObj<TNum>& omega_y, VectorObj<TNum>& omega_z) const {
        // Resize output arrays if needed
        if (omega_x.size() != nx * ny * nz) {
            omega_x.resize(nx * ny * nz);
            omega_y.resize(nx * ny * nz);
            omega_z.resize(nx * ny * nz);
        }
        
        // Calculate vorticity in spectral space
       VectorObj<std::complex<TNum>> omega_x_hat(nx * ny * nz);
       VectorObj<std::complex<TNum>> omega_y_hat(nx * ny * nz);
       VectorObj<std::complex<TNum>> omega_z_hat(nx * ny * nz);
        
        std::complex<TNum> I(0, 1);
        
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    int index = idx(i, j, k);
                    
                    // ω_x = ∂w/∂y - ∂v/∂z
                    omega_x_hat[index] = I * (ky[j] * w_hat[index] - kz[k] * v_hat[index]);
                    
                    // ω_y = ∂u/∂z - ∂w/∂x
                    omega_y_hat[index] = I * (kz[k] * u_hat[index] - kx[i] * w_hat[index]);
                    
                    // ω_z = ∂v/∂x - ∂u/∂y
                    omega_z_hat[index] = I * (kx[i] * v_hat[index] - ky[j] * u_hat[index]);
                }
            }
        }
        
        // Transform vorticity to physical space
        fftw_complex *temp_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
        fftw_complex *temp_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
        fftw_plan temp_plan = fftw_plan_dft_3d(nx, ny, nz, temp_in, temp_out, FFTW_BACKWARD, FFTW_ESTIMATE);
        
        // Transform omega_x
        for (int i = 0; i < nx * ny * nz; i++) {
            temp_in[i][0] = omega_x_hat[i].real();
            temp_in[i][1] = omega_x_hat[i].imag();
        }
        fftw_execute(temp_plan);
        for (int i = 0; i < nx * ny * nz; i++) {
            omega_x[i] = temp_out[i][0];
        }
        
        // Transform omega_y
        for (int i = 0; i < nx * ny * nz; i++) {
            temp_in[i][0] = omega_y_hat[i].real();
            temp_in[i][1] = omega_y_hat[i].imag();
        }
        fftw_execute(temp_plan);
        for (int i = 0; i < nx * ny * nz; i++) {
            omega_y[i] = temp_out[i][0];
        }
        
        // Transform omega_z
        for (int i = 0; i < nx * ny * nz; i++) {
            temp_in[i][0] = omega_z_hat[i].real();
            temp_in[i][1] = omega_z_hat[i].imag();
        }
        fftw_execute(temp_plan);
        for (int i = 0; i < nx * ny * nz; i++) {
            omega_z[i] = temp_out[i][0];
        }
        
        // Clean up
        fftw_destroy_plan(temp_plan);
        fftw_free(temp_in);
        fftw_free(temp_out);
    }
    
    // Get pressure field
    VectorObj<TNum> getPressureField() {
        // Compute pressure in spectral space
        computePressure();
        
        // Transform pressure to physical space
        VectorObj<TNum> p_physical(nx * ny * nz);
        
        fftw_complex *temp_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
        fftw_complex *temp_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * nz);
        fftw_plan temp_plan = fftw_plan_dft_3d(nx, ny, nz, temp_in, temp_out, FFTW_BACKWARD, FFTW_ESTIMATE);
        
        // Copy pressure to FFTW array
        for (int i = 0; i < nx * ny * nz; i++) {
            temp_in[i][0] = p_hat[i].real();
            temp_in[i][1] = p_hat[i].imag();
        }
        
        // Execute FFT
        fftw_execute(temp_plan);
        
        // Copy result to pressure array
        for (int i = 0; i < nx * ny * nz; i++) {
            p_physical[i] = temp_out[i][0];
        }
        
        // Clean up
        fftw_destroy_plan(temp_plan);
        fftw_free(temp_in);
        fftw_free(temp_out);
        
        return p_physical;
    }
    
    // Set adaptive time stepping
    void enableAdaptiveTimeStep(TNum target_cfl = 0.5) {
        adaptive_dt = true;
        cfl_target = target_cfl;
    }
    
    void disableAdaptiveTimeStep() {
        adaptive_dt = false;
    }
    
    // Adjust time step based on CFL condition
    void adjustTimeStep() {
        if (!adaptive_dt) return;
        
        TNum current_cfl = calculateCFL();
        
        if (current_cfl > 0) {
            // Scale dt to match target CFL
            dt = dt * (cfl_target / current_cfl);
            
            // Add safety factor
            dt *= 0.9;
        }
    }

    
private:
    // Adaptive time stepping parameters
    bool adaptive_dt = false;
    TNum cfl_target = 0.5;
};


#endif // SPECTRAL_METHOD_HPP
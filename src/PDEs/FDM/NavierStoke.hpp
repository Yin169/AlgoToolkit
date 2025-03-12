#ifndef NAVIER_STOKES_HPP
#define NAVIER_STOKES_HPP

#include "../../Obj/VectorObj.hpp"
#include "../../Obj/SparseObj.hpp"
#include "../../Obj/DenseObj.hpp"
#include "../../LinearAlgebra/Preconditioner/MultiGrid.hpp"
#include "../../ODE/RungeKutta.hpp"
#include "../FFT/FastPoisson.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <functional>
#include <algorithm>
#include <string>
#include <memory>

enum class TimeIntegration {
    EULER,
    RK4
};

enum class AdvectionScheme {
    UPWIND,
    CENTRAL,
    QUICK
};

template <typename TNum>
class NavierStokesSolver3D {
private:
    // Grid dimensions
    int nx, ny, nz;
    TNum dx, dy, dz;
    TNum dt;
    TNum Re;  // Reynolds number
    TNum cfl_target = 0.5; // Target CFL number for adaptive time stepping
    bool adaptive_dt = false; // Flag for adaptive time stepping
    
    // Velocity components (u, v, w) and pressure (p)
    VectorObj<TNum> u, v, w, p;
    VectorObj<TNum> u_prev, v_prev, w_prev;
    
    // External forces
    VectorObj<TNum> fx, fy, fz;
    bool has_external_forces = false;
    
    // System matrices for pressure Poisson equation
    SparseMatrixCSC<TNum> laplacian;
    AlgebraicMultiGrid<TNum, VectorObj<TNum>> amg;
    
    // Fast Poisson solver using FFT
    std::unique_ptr<FastPoisson3D<TNum>> fast_poisson;
    bool use_fft_solver = false;
    
    TimeIntegration time_method = TimeIntegration::RK4;
    std::unique_ptr<RungeKutta<TNum, VectorObj<TNum>>> rk_solver;
    
    AdvectionScheme advection_scheme = AdvectionScheme::UPWIND;
    
    // Boundary conditions
    std::function<TNum(TNum, TNum, TNum, TNum)> u_bc, v_bc, w_bc;
    
    // Helper methods
    int idx(int i, int j, int k) const {
        return i + j * nx + k * nx * ny;
    }
    
    void buildLaplacianMatrix() {
        int n = nx * ny * nz;
        laplacian = SparseMatrixCSC<TNum>(n, n);
        
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    int index = idx(i, j, k);
                    
                    // Diagonal term
                    TNum diag = -6.0;
                    
                    // Boundary adjustments
                    if (i == 0 || i == nx-1) diag += 1.0;
                    if (j == 0 || j == ny-1) diag += 1.0;
                    if (k == 0 || k == nz-1) diag += 1.0;
                    
                    laplacian.addValue(index, index, diag / (dx * dx));
                    
                    // Off-diagonal terms
                    if (i > 0) laplacian.addValue(index, idx(i-1, j, k), 1.0 / (dx * dx));
                    if (i < nx-1) laplacian.addValue(index, idx(i+1, j, k), 1.0 / (dx * dx));
                    if (j > 0) laplacian.addValue(index, idx(i, j-1, k), 1.0 / (dy * dy));
                    if (j < ny-1) laplacian.addValue(index, idx(i, j+1, k), 1.0 / (dy * dy));
                    if (k > 0) laplacian.addValue(index, idx(i, j, k-1), 1.0 / (dz * dz));
                    if (k < nz-1) laplacian.addValue(index, idx(i, j, k+1), 1.0 / (dz * dz));
                }
            }
        }
        
        laplacian.finalize();
    }
    
    void applyBoundaryConditions(TNum time) {
        // Apply boundary conditions to velocity fields
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                // x boundaries
                u[idx(0, j, k)] = u_bc(0, j*dy, k*dz, time);
                u[idx(nx-1, j, k)] = u_bc((nx-1)*dx, j*dy, k*dz, time);
                
                // y boundaries
                v[idx(0, j, k)] = v_bc(0, j*dy, k*dz, time);
                v[idx(nx-1, j, k)] = v_bc((nx-1)*dx, j*dy, k*dz, time);
                
                // z boundaries
                w[idx(0, j, k)] = w_bc(0, j*dy, k*dz, time);
                w[idx(nx-1, j, k)] = w_bc((nx-1)*dx, j*dy, k*dz, time);
            }
        }
        
        for (int k = 0; k < nz; k++) {
            for (int i = 0; i < nx; i++) {
                // x boundaries
                u[idx(i, 0, k)] = u_bc(i*dx, 0, k*dz, time);
                u[idx(i, ny-1, k)] = u_bc(i*dx, (ny-1)*dy, k*dz, time);
                
                // y boundaries
                v[idx(i, 0, k)] = v_bc(i*dx, 0, k*dz, time);
                v[idx(i, ny-1, k)] = v_bc(i*dx, (ny-1)*dy, k*dz, time);
                
                // z boundaries
                w[idx(i, 0, k)] = w_bc(i*dx, 0, k*dz, time);
                w[idx(i, ny-1, k)] = w_bc(i*dx, (ny-1)*dy, k*dz, time);
            }
        }
        
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                // x boundaries
                u[idx(i, j, 0)] = u_bc(i*dx, j*dy, 0, time);
                u[idx(i, j, nz-1)] = u_bc(i*dx, j*dy, (nz-1)*dz, time);
                
                // y boundaries
                v[idx(i, j, 0)] = v_bc(i*dx, j*dy, 0, time);
                v[idx(i, j, nz-1)] = v_bc(i*dx, j*dy, (nz-1)*dz, time);
                
                // z boundaries
                w[idx(i, j, 0)] = w_bc(i*dx, j*dy, 0, time);
                w[idx(i, j, nz-1)] = w_bc(i*dx, j*dy, (nz-1)*dz, time);
            }
        }
    }
    
    // Compute advection terms using the selected scheme
    void computeAdvectionTerms(int i, int j, int k, TNum& u_adv, TNum& v_adv, TNum& w_adv) {
        int idx_c = idx(i, j, k);
        
        switch (advection_scheme) {
            case AdvectionScheme::UPWIND:
                computeUpwindAdvection(i, j, k, idx_c, u_adv, v_adv, w_adv);
                break;
            case AdvectionScheme::CENTRAL:
                computeCentralAdvection(i, j, k, idx_c, u_adv, v_adv, w_adv);
                break;
            case AdvectionScheme::QUICK:
                computeQUICKAdvection(i, j, k, idx_c, u_adv, v_adv, w_adv);
                break;
        }
    }
    
    // First-order upwind scheme for advection
    void computeUpwindAdvection(int i, int j, int k, int idx_c, TNum& u_adv, TNum& v_adv, TNum& w_adv) {
        // u-component advection
        if (u[idx_c] > 0) {
            u_adv = u[idx_c] * (u[idx_c] - u[idx(i-1, j, k)]) / dx;
        } else {
            u_adv = u[idx_c] * (u[idx(i+1, j, k)] - u[idx_c]) / dx;
        }
        
        if (v[idx_c] > 0) {
            u_adv += v[idx_c] * (u[idx_c] - u[idx(i, j-1, k)]) / dy;
        } else {
            u_adv += v[idx_c] * (u[idx(i, j+1, k)] - u[idx_c]) / dy;
        }
        
        if (w[idx_c] > 0) {
            u_adv += w[idx_c] * (u[idx_c] - u[idx(i, j, k-1)]) / dz;
        } else {
            u_adv += w[idx_c] * (u[idx(i, j, k+1)] - u[idx_c]) / dz;
        }
        
        // v-component advection
        if (u[idx_c] > 0) {
            v_adv = u[idx_c] * (v[idx_c] - v[idx(i-1, j, k)]) / dx;
        } else {
            v_adv = u[idx_c] * (v[idx(i+1, j, k)] - v[idx_c]) / dx;
        }
        
        if (v[idx_c] > 0) {
            v_adv += v[idx_c] * (v[idx_c] - v[idx(i, j-1, k)]) / dy;
        } else {
            v_adv += v[idx_c] * (v[idx(i, j+1, k)] - v[idx_c]) / dy;
        }
        
        if (w[idx_c] > 0) {
            v_adv += w[idx_c] * (v[idx_c] - v[idx(i, j, k-1)]) / dz;
        } else {
            v_adv += w[idx_c] * (v[idx(i, j, k+1)] - v[idx_c]) / dz;
        }
        
        // w-component advection
        if (u[idx_c] > 0) {
            w_adv = u[idx_c] * (w[idx_c] - w[idx(i-1, j, k)]) / dx;
        } else {
            w_adv = u[idx_c] * (w[idx(i+1, j, k)] - w[idx_c]) / dx;
        }
        
        if (v[idx_c] > 0) {
            w_adv += v[idx_c] * (w[idx_c] - w[idx(i, j-1, k)]) / dy;
        } else {
            w_adv += v[idx_c] * (w[idx(i, j+1, k)] - w[idx_c]) / dy;
        }
        
        if (w[idx_c] > 0) {
            w_adv += w[idx_c] * (w[idx_c] - w[idx(i, j, k-1)]) / dz;
        } else {
            w_adv += w[idx_c] * (w[idx(i, j, k+1)] - w[idx_c]) / dz;
        }
    }
    
    // Second-order central difference scheme for advection
    void computeCentralAdvection(int i, int j, int k, int idx_c, TNum& u_adv, TNum& v_adv, TNum& w_adv) {
        // u-component advection
        u_adv = u[idx_c] * (u[idx(i+1, j, k)] - u[idx(i-1, j, k)]) / (2*dx) +
                v[idx_c] * (u[idx(i, j+1, k)] - u[idx(i, j-1, k)]) / (2*dy) +
                w[idx_c] * (u[idx(i, j, k+1)] - u[idx(i, j, k-1)]) / (2*dz);
        
        // v-component advection
        v_adv = u[idx_c] * (v[idx(i+1, j, k)] - v[idx(i-1, j, k)]) / (2*dx) +
                v[idx_c] * (v[idx(i, j+1, k)] - v[idx(i, j-1, k)]) / (2*dy) +
                w[idx_c] * (v[idx(i, j, k+1)] - v[idx(i, j, k-1)]) / (2*dz);
        
        // w-component advection
        w_adv = u[idx_c] * (w[idx(i+1, j, k)] - w[idx(i-1, j, k)]) / (2*dx) +
                v[idx_c] * (w[idx(i, j+1, k)] - w[idx(i, j-1, k)]) / (2*dy) +
                w[idx_c] * (w[idx(i, j, k+1)] - w[idx(i, j, k-1)]) / (2*dz);
    }
    
    // QUICK (Quadratic Upstream Interpolation for Convective Kinematics) scheme
    void computeQUICKAdvection(int i, int j, int k, int idx_c, TNum& u_adv, TNum& v_adv, TNum& w_adv) {
        // Only implement if we have enough grid points in each direction
        if (nx < 4 || ny < 4 || nz < 4) {
            computeCentralAdvection(i, j, k, idx_c, u_adv, v_adv, w_adv);
            return;
        }
        
        // Implement QUICK scheme for u-component in x-direction as an example
        // For brevity, we'll use central differences for other components
        
        // u-component advection in x-direction with QUICK
        if (i > 1 && i < nx-2) {
            if (u[idx_c] >= 0) {
                // QUICK formula for positive velocity: φ_i+1/2 = 1/8(3φ_i+1 + 6φ_i - φ_i-1)
                u_adv = u[idx_c] * (3*u[idx(i+1, j, k)] + 6*u[idx_c] - u[idx(i-1, j, k)]) / (8*dx);
            } else {
                // QUICK formula for negative velocity: φ_i-1/2 = 1/8(3φ_i-1 + 6φ_i - φ_i+1)
                u_adv = u[idx_c] * (3*u[idx(i-1, j, k)] + 6*u[idx_c] - u[idx(i+1, j, k)]) / (8*dx);
            }
        } else {
            // Fall back to central difference near boundaries
            u_adv = u[idx_c] * (u[idx(i+1, j, k)] - u[idx(i-1, j, k)]) / (2*dx);
        }
        
        // Use central differences for other directions for simplicity
        u_adv += v[idx_c] * (u[idx(i, j+1, k)] - u[idx(i, j-1, k)]) / (2*dy) +
                 w[idx_c] * (u[idx(i, j, k+1)] - u[idx(i, j, k-1)]) / (2*dz);
        
        // Use central differences for v and w components
        v_adv = u[idx_c] * (v[idx(i+1, j, k)] - v[idx(i-1, j, k)]) / (2*dx) +
                v[idx_c] * (v[idx(i, j+1, k)] - v[idx(i, j-1, k)]) / (2*dy) +
                w[idx_c] * (v[idx(i, j, k+1)] - v[idx(i, j, k-1)]) / (2*dz);
        
        w_adv = u[idx_c] * (w[idx(i+1, j, k)] - w[idx(i-1, j, k)]) / (2*dx) +
                v[idx_c] * (w[idx(i, j+1, k)] - w[idx(i, j-1, k)]) / (2*dy) +
                w[idx_c] * (w[idx(i, j, k+1)] - w[idx(i, j, k-1)]) / (2*dz);
    }
    
    // Calculate diffusion terms using central differences
    void computeDiffusionTerms(int i, int j, int k, TNum& u_diff, TNum& v_diff, TNum& w_diff) {
        int idx_c = idx(i, j, k);
        
        // u-component diffusion
        u_diff = (u[idx(i+1, j, k)] - 2*u[idx_c] + u[idx(i-1, j, k)]) / (dx*dx) +
                 (u[idx(i, j+1, k)] - 2*u[idx_c] + u[idx(i, j-1, k)]) / (dy*dy) +
                 (u[idx(i, j, k+1)] - 2*u[idx_c] + u[idx(i, j, k-1)]) / (dz*dz);
        
        // v-component diffusion
        v_diff = (v[idx(i+1, j, k)] - 2*v[idx_c] + v[idx(i-1, j, k)]) / (dx*dx) +
                 (v[idx(i, j+1, k)] - 2*v[idx_c] + v[idx(i, j-1, k)]) / (dy*dy) +
                 (v[idx(i, j, k+1)] - 2*v[idx_c] + v[idx(i, j, k-1)]) / (dz*dz);
        
        // w-component diffusion
        w_diff = (w[idx(i+1, j, k)] - 2*w[idx_c] + w[idx(i-1, j, k)]) / (dx*dx) +
                 (w[idx(i, j+1, k)] - 2*w[idx_c] + w[idx(i, j-1, k)]) / (dy*dy) +
                 (w[idx(i, j, k+1)] - 2*w[idx_c] + w[idx(i, j, k-1)]) / (dz*dz);
    }
    
    // Compute the right-hand side for the momentum equations
    VectorObj<TNum> computeMomentumRHS(const VectorObj<TNum>& velocity, int component) {
        int n = nx * ny * nz;
        VectorObj<TNum> rhs(n, 0.0);
        
        for (int k = 1; k < nz-1; k++) {
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    int idx_c = idx(i, j, k);
                    
                    TNum u_adv = 0, v_adv = 0, w_adv = 0;
                    TNum u_diff = 0, v_diff = 0, w_diff = 0;
                    
                    computeAdvectionTerms(i, j, k, u_adv, v_adv, w_adv);
                    computeDiffusionTerms(i, j, k, u_diff, v_diff, w_diff);
                    
                    // Add external forces if present
                    TNum force_term = 0.0;
                    if (has_external_forces) {
                        if (component == 0) force_term = fx[idx_c];
                        else if (component == 1) force_term = fy[idx_c];
                        else force_term = fz[idx_c];
                    }
                    
                    // Set RHS based on component
                    if (component == 0) {
                        rhs[idx_c] = -u_adv + (1.0/Re) * u_diff + force_term;
                    } else if (component == 1) {
                        rhs[idx_c] = -v_adv + (1.0/Re) * v_diff + force_term;
                    } else {
                        rhs[idx_c] = -w_adv + (1.0/Re) * w_diff + force_term;
                    }
                }
            }
        }
        
        return rhs;
    }
    
    // Compute intermediate velocity using Euler method
    void computeIntermediateVelocityEuler() {
        // Store previous velocity
        u_prev = u;
        v_prev = v;
        w_prev = w;
        
        // Compute intermediate velocity field using explicit Euler scheme
        for (int k = 1; k < nz-1; k++) {
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    int idx_c = idx(i, j, k);
                    
                    // Advection terms
                    TNum u_adv = 0, v_adv = 0, w_adv = 0;
                    
                    // Diffusion terms
                    TNum u_diff = 0, v_diff = 0, w_diff = 0;
                    
                    computeAdvectionTerms(i, j, k, u_adv, v_adv, w_adv);
                    computeDiffusionTerms(i, j, k, u_diff, v_diff, w_diff);
                    
                    // External forces
                    TNum fx_term = has_external_forces ? fx[idx_c] : 0.0;
                    TNum fy_term = has_external_forces ? fy[idx_c] : 0.0;
                    TNum fz_term = has_external_forces ? fz[idx_c] : 0.0;
                    
                    // Update intermediate velocity
                    u[idx_c] = u_prev[idx_c] + dt * (-u_adv + (1.0/Re) * u_diff + fx_term);
                    v[idx_c] = v_prev[idx_c] + dt * (-v_adv + (1.0/Re) * v_diff + fy_term);
                    w[idx_c] = w_prev[idx_c] + dt * (-w_adv + (1.0/Re) * w_diff + fz_term);
                }
            }
        }
    }
    
    // Compute intermediate velocity using RK4 method
    void computeIntermediateVelocityRK4() {
        // Create system functions for each velocity component
        auto u_system = [this](const VectorObj<TNum>& u_vec) -> VectorObj<TNum> {
            // Store current u for computation
            VectorObj<TNum> u_temp = u;
            u = u_vec;
            VectorObj<TNum> result = computeMomentumRHS(u, 0);
            u = u_temp; // Restore original u
            return result;
        };
        
        auto v_system = [this](const VectorObj<TNum>& v_vec) -> VectorObj<TNum> {
            // Store current v for computation
            VectorObj<TNum> v_temp = v;
            v = v_vec;
            VectorObj<TNum> result = computeMomentumRHS(v, 1);
            v = v_temp; // Restore original v
            return result;
        };
        
        auto w_system = [this](const VectorObj<TNum>& w_vec) -> VectorObj<TNum> {
            // Store current w for computation
            VectorObj<TNum> w_temp = w;
            w = w_vec;
            VectorObj<TNum> result = computeMomentumRHS(w, 2);
            w = w_temp; // Restore original w
            return result;
        };
        
        // Store previous velocity
        u_prev = u;
        v_prev = v;
        w_prev = w;
        
        // Solve using RK4
        rk_solver->solve(u, u_system, dt, 1);
        rk_solver->solve(v, v_system, dt, 1);
        rk_solver->solve(w, w_system, dt, 1);
    }
    
    void computeIntermediateVelocity() {
        if (time_method == TimeIntegration::EULER) {
            computeIntermediateVelocityEuler();
        } else {
            computeIntermediateVelocityRK4();
        }
    }
    
    // Modified solvePressurePoisson method to use either AMG or FFT
    void solvePressurePoisson() {
        int n = nx * ny * nz;
        VectorObj<TNum> rhs(n, 0.0);
        
        // Compute right-hand side of pressure Poisson equation
        for (int k = 1; k < nz-1; k++) {
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    int idx_c = idx(i, j, k);
                    
                    rhs[idx_c] = (1.0/dt) * (
                        (u[idx(i+1, j, k)] - u[idx(i-1, j, k)]) / (2*dx) +
                        (v[idx(i, j+1, k)] - v[idx(i, j-1, k)]) / (2*dy) +
                        (w[idx(i, j, k+1)] - w[idx(i, j, k-1)]) / (2*dz)
                    );
                }
            }
        }
        
        // Apply Neumann boundary conditions for pressure (dp/dn = 0)
        // This ensures that the normal pressure gradient at boundaries is zero
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                // x boundaries
                rhs[idx(0, j, k)] = 0.0;
                rhs[idx(nx-1, j, k)] = 0.0;
            }
        }
        
        for (int k = 0; k < nz; k++) {
            for (int i = 0; i < nx; i++) {
                // y boundaries
                rhs[idx(i, 0, k)] = 0.0;
                rhs[idx(i, ny-1, k)] = 0.0;
            }
        }
        
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                // z boundaries
                rhs[idx(i, j, 0)] = 0.0;
                rhs[idx(i, j, nz-1)] = 0.0;
            }
        }
        
        // Initialize pressure to zero before solving
        p.zero();
        
        if (use_fft_solver && fast_poisson) {
            // Solve using FFT-based Poisson solver
            p = fast_poisson->solve(rhs);
        } else {
            // Solve using algebraic multigrid
            amg.amgVCycle(laplacian, rhs, p, 3, 5, 0.25);
        }
    }
    
    void projectVelocity() {
        // Project velocity field to ensure incompressibility
        for (int k = 1; k < nz-1; k++) {
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    int idx_c = idx(i, j, k);
                    
                    u[idx_c] -= dt * (p[idx(i+1, j, k)] - p[idx(i-1, j, k)]) / (2*dx);
                    v[idx_c] -= dt * (p[idx(i, j+1, k)] - p[idx(i, j-1, k)]) / (2*dy);
                    w[idx_c] -= dt * (p[idx(i, j, k+1)] - p[idx(i, j, k-1)]) / (2*dz);
                }
            }
        }
    }
    
    // Calculate maximum CFL number for adaptive time stepping
    TNum calculateMaxCFL() const {
        TNum max_vel = 0.0;
        
        for (int k = 1; k < nz-1; k++) {
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    int index = idx(i, j, k);
                    TNum vel_mag = std::sqrt(u[index]*u[index] + v[index]*v[index] + w[index]*w[index]);
                    max_vel = std::max(max_vel, vel_mag);
                }
            }
        }
        
        // Calculate CFL number: max_vel * dt / min_dx
        TNum min_dx = std::min(std::min(dx, dy), dz);
        return max_vel * dt / min_dx;
    }
    
    // Adjust time step based on CFL condition
    // Adjust time step based on CFL condition
    void adjustTimeStep() {
        if (!adaptive_dt) return;
        
        TNum current_cfl = calculateMaxCFL();
        
        if (current_cfl > 0) {
            // Adjust dt to match target CFL
            dt = cfl_target * dt / current_cfl;
            
            // Add safety factor and limit maximum change
            dt *= 0.9;  // Safety factor
            
            // Limit maximum increase to 1.5x
            if (dt > 1.5 * dt) dt = 1.5 * dt;
        }
    }

public:
    // Constructor
    NavierStokesSolver3D(int nx_, int ny_, int nz_, TNum dx_, TNum dy_, TNum dz_, TNum dt_, TNum Re_) 
        : nx(nx_), ny(ny_), nz(nz_), dx(dx_), dy(dy_), dz(dz_), dt(dt_), Re(Re_) {
        
        // Initialize velocity and pressure fields
        int n = nx * ny * nz;
        u.resize(n, 0.0);
        v.resize(n, 0.0);
        w.resize(n, 0.0);
        p.resize(n, 0.0);
        
        u_prev.resize(n, 0.0);
        v_prev.resize(n, 0.0);
        w_prev.resize(n, 0.0);
        
        // Initialize external forces
        fx.resize(n, 0.0);
        fy.resize(n, 0.0);
        fz.resize(n, 0.0);
        
        // Build Laplacian matrix for pressure Poisson equation
        buildLaplacianMatrix();
        
        // Initialize RK4 solver
        rk_solver = std::make_unique<RungeKutta<TNum, VectorObj<TNum>>>();
        fast_poisson = std::make_unique<FastPoisson3D<TNum>>(nx, ny, nz, nx*dx, ny*dy, nz*dz);
        
        
        // Default boundary conditions (zero velocity)
        u_bc = [](TNum x, TNum y, TNum z, TNum t) { return 0.0; };
        v_bc = [](TNum x, TNum y, TNum z, TNum t) { return 0.0; };
        w_bc = [](TNum x, TNum y, TNum z, TNum t) { return 0.0; };
    }
    
    // Set boundary conditions
    void setBoundaryConditions(
        std::function<TNum(TNum, TNum, TNum, TNum)> u_bc_,
        std::function<TNum(TNum, TNum, TNum, TNum)> v_bc_,
        std::function<TNum(TNum, TNum, TNum, TNum)> w_bc_) {
        u_bc = u_bc_;
        v_bc = v_bc_;
        w_bc = w_bc_;
    }
    
    // Set external forces
    void setExternalForces(const VectorObj<TNum>& fx_, const VectorObj<TNum>& fy_, const VectorObj<TNum>& fz_) {
        fx = fx_;
        fy = fy_;
        fz = fz_;
        has_external_forces = true;
    }
    
    // Set initial conditions
    void setInitialConditions(const VectorObj<TNum>& u0, const VectorObj<TNum>& v0, const VectorObj<TNum>& w0) {
        u = u0;
        v = v0;
        w = w0;
    }
   
    void setFFTPoissonSolver(){
        use_fft_solver = true;
    }

    // Set time integration method
    void setTimeIntegrationMethod(TimeIntegration method) {
        time_method = method;
    }
    
    // Set advection scheme
    void setAdvectionScheme(AdvectionScheme scheme) {
        advection_scheme = scheme;
    }
    
    // Enable adaptive time stepping
    void enableAdaptiveTimeStep(TNum cfl_target_) {
        adaptive_dt = true;
        cfl_target = cfl_target_;
    }
    
    // Disable adaptive time stepping
    void disableAdaptiveTimeStep() {
        adaptive_dt = false;
    }
    
    // Solve for a single time step
    void step(TNum time) {
        // Apply boundary conditions
        applyBoundaryConditions(time);
        
        // Compute intermediate velocity field (without pressure gradient)
        computeIntermediateVelocity();
        
        // Solve pressure Poisson equation
        solvePressurePoisson();
        
        // Project velocity field to ensure incompressibility
        projectVelocity();
        
        // Apply boundary conditions again
        applyBoundaryConditions(time);
        
        // Adjust time step if adaptive time stepping is enabled
        adjustTimeStep();
    }
    
    // Solve for multiple time steps
    void solve(TNum start_time, TNum end_time, int output_frequency = 1,
               std::function<void(const VectorObj<TNum>&, const VectorObj<TNum>&, 
                                 const VectorObj<TNum>&, const VectorObj<TNum>&, TNum)> output_callback = nullptr) {
        TNum time = start_time;
        int step_count = 0;
        
        while (time < end_time) {
            // Output results if needed
            if (output_callback && step_count % output_frequency == 0) {
                output_callback(u, v, w, p, time);
            }
            // Solve for a single time step
            step(time);
            
            // Update time
            time += dt;
            step_count++;
            
        }
    }
    
    // Get velocity and pressure fields
    const VectorObj<TNum>& getU() const { return u; }
    const VectorObj<TNum>& getV() const { return v; }
    const VectorObj<TNum>& getW() const { return w; }
    const VectorObj<TNum>& getP() const { return p; }
    
    // Get grid dimensions
    int getNx() const { return nx; }
    int getNy() const { return ny; }
    int getNz() const { return nz; }
    
    // Get grid spacing
    TNum getDx() const { return dx; }
    TNum getDy() const { return dy; }
    TNum getDz() const { return dz; }
    
    // Get time step
    TNum getDt() const { return dt; }
    
    // Set time step
    void setDt(TNum dt_) { dt = dt_; }
    
    // Get Reynolds number
    TNum getReynoldsNumber() const { return Re; }
    
    // Calculate divergence of velocity field (for debugging)
    VectorObj<TNum> calculateDivergence() const {
        int n = nx * ny * nz;
        VectorObj<TNum> div(n, 0.0);
        
        for (int k = 1; k < nz-1; k++) {
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    int idx_c = idx(i, j, k);
                    
                    div[idx_c] = (u[idx(i+1, j, k)] - u[idx(i-1, j, k)]) / (2*dx) +
                                 (v[idx(i, j+1, k)] - v[idx(i, j-1, k)]) / (2*dy) +
                                 (w[idx(i, j, k+1)] - w[idx(i, j, k-1)]) / (2*dz);
                }
            }
        }
        
        return div;
    }
    
    // Calculate vorticity field
    std::tuple<VectorObj<TNum>, VectorObj<TNum>, VectorObj<TNum>> calculateVorticity() const {
        int n = nx * ny * nz;
        VectorObj<TNum> vort_x(n, 0.0);
        VectorObj<TNum> vort_y(n, 0.0);
        VectorObj<TNum> vort_z(n, 0.0);
        
        for (int k = 1; k < nz-1; k++) {
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    int idx_c = idx(i, j, k);
                    
                    // ω_x = ∂w/∂y - ∂v/∂z
                    vort_x[idx_c] = (w[idx(i, j+1, k)] - w[idx(i, j-1, k)]) / (2*dy) -
                                    (v[idx(i, j, k+1)] - v[idx(i, j, k-1)]) / (2*dz);
                    
                    // ω_y = ∂u/∂z - ∂w/∂x
                    vort_y[idx_c] = (u[idx(i, j, k+1)] - u[idx(i, j, k-1)]) / (2*dz) -
                                    (w[idx(i+1, j, k)] - w[idx(i-1, j, k)]) / (2*dx);
                    
                    // ω_z = ∂v/∂x - ∂u/∂y
                    vort_z[idx_c] = (v[idx(i+1, j, k)] - v[idx(i-1, j, k)]) / (2*dx) -
                                    (u[idx(i, j+1, k)] - u[idx(i, j-1, k)]) / (2*dy);
                }
            }
        }
        
        return std::make_tuple(vort_x, vort_y, vort_z);
    }
    
    // Calculate kinetic energy
    TNum calculateKineticEnergy() const {
        TNum energy = 0.0;
        
        for (int k = 1; k < nz-1; k++) {
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    int idx_c = idx(i, j, k);
                    energy += 0.5 * (u[idx_c]*u[idx_c] + v[idx_c]*v[idx_c] + w[idx_c]*w[idx_c]);
                }
            }
        }
        
        return energy * dx * dy * dz;
    }
};

#endif // NAVIER_STOKES_HPP
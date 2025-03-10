#ifndef NAVIER_STOKES_HPP
#define NAVIER_STOKES_HPP

#include "../../Obj/VectorObj.hpp"
#include "../../Obj/SparseObj.hpp"
#include "../../Obj/DenseObj.hpp"
#include "../../LinearAlgebra/Preconditioner/MultiGrid.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <functional>
#include <algorithm>

template <typename TNum>
class NavierStokesSolver3D {
private:
    // Grid dimensions
    int nx, ny, nz;
    TNum dx, dy, dz;
    TNum dt, dt_max;
    TNum Re;  // Reynolds number
    TNum cfl; // CFL number for adaptive time stepping
    bool adaptive_dt; // Flag for adaptive time stepping
    
    // Turbulence modeling
    bool use_turbulence_model;
    TNum Cs; // Smagorinsky constant
    
    // Pressure solver parameters
    int pressure_iterations;
    TNum pressure_tolerance;
    
    // Velocity components (u, v, w) and pressure (p)
    VectorObj<TNum> u, v, w, p;
    VectorObj<TNum> u_prev, v_prev, w_prev;
    
    // External forces
    VectorObj<TNum> fx, fy, fz;
    
    // Turbulent viscosity for LES model
    VectorObj<TNum> nu_t;
    
    // System matrices for pressure Poisson equation
    SparseMatrixCSC<TNum> laplacian;
    AlgebraicMultiGrid<TNum, VectorObj<TNum>> amg;
    
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
    
    TNum wenoDerivative(TNum v1, TNum v2, TNum v3, TNum v4, TNum v5, TNum h, bool positive) {
        // WENO (Weighted Essentially Non-Oscillatory) scheme for high-order advection
        const TNum epsilon = 1e-6; // Small value to avoid division by zero
        
        // Compute smoothness indicators
        TNum beta1 = std::pow(v1 - 2*v2 + v3, 2) + 0.25 * std::pow(v1 - 4*v2 + 3*v3, 2);
        TNum beta2 = std::pow(v2 - 2*v3 + v4, 2) + 0.25 * std::pow(v2 - v4, 2);
        TNum beta3 = std::pow(v3 - 2*v4 + v5, 2) + 0.25 * std::pow(3*v3 - 4*v4 + v5, 2);
        
        // Compute nonlinear weights
        TNum alpha1 = 0.1 / std::pow(epsilon + beta1, 2);
        TNum alpha2 = 0.6 / std::pow(epsilon + beta2, 2);
        TNum alpha3 = 0.3 / std::pow(epsilon + beta3, 2);
        
        TNum sum_alpha = alpha1 + alpha2 + alpha3;
        TNum w1 = alpha1 / sum_alpha;
        TNum w2 = alpha2 / sum_alpha;
        TNum w3 = alpha3 / sum_alpha;
        
        // Compute stencil approximations
        TNum s1, s2, s3;
        if (positive) {
            s1 = (1.0/6.0) * (2*v1 - 7*v2 + 11*v3) / h;
            s2 = (1.0/6.0) * (-v2 + 5*v3 + 2*v4) / h;
            s3 = (1.0/6.0) * (2*v3 + 5*v4 - v5) / h;
        } else {
            s1 = (1.0/6.0) * (-v1 + 5*v2 + 2*v3) / h;
            s2 = (1.0/6.0) * (2*v2 + 5*v3 - v4) / h;
            s3 = (1.0/6.0) * (11*v3 - 7*v4 + 2*v5) / h;
        }
        
        // Combine stencils with nonlinear weights
        return w1 * s1 + w2 * s2 + w3 * s3;
    }
    
    TNum computeAdvection(const VectorObj<TNum>& vel_field, int i, int j, int k, int component) {
        // WENO (Weighted Essentially Non-Oscillatory) scheme for advection
        TNum adv_term = 0.0;
        int idx_c = idx(i, j, k);
        
        // Check if we have enough points for WENO stencil in each direction
        bool has_x_stencil = (i >= 2 && i < nx-2);
        bool has_y_stencil = (j >= 2 && j < ny-2);
        bool has_z_stencil = (k >= 2 && k < nz-2);
        
        // x-direction advection
        TNum u_face = 0.5 * (u[idx_c] + u[idx(i+1 < nx ? i+1 : i, j, k)]);
        if (has_x_stencil) {
            if (u_face > 0) {
                // Upwind stencil for positive velocity
                adv_term += u_face * wenoDerivative(
                    vel_field[idx(i-2, j, k)], 
                    vel_field[idx(i-1, j, k)], 
                    vel_field[idx_c], 
                    vel_field[idx(i+1, j, k)], 
                    vel_field[idx(i+2, j, k)], 
                    dx, true);
            } else {
                // Upwind stencil for negative velocity
                adv_term += u_face * wenoDerivative(
                    vel_field[idx(i+2, j, k)], 
                    vel_field[idx(i+1, j, k)], 
                    vel_field[idx_c], 
                    vel_field[idx(i-1, j, k)], 
                    vel_field[idx(i-2, j, k)], 
                    dx, false);
            }
        } else {
            // Fall back to first-order upwind near boundaries
            if (u_face > 0 && i > 0) {
                adv_term += u_face * (vel_field[idx_c] - vel_field[idx(i-1, j, k)]) / dx;
            } else if (u_face < 0 && i < nx-1) {
                adv_term += u_face * (vel_field[idx(i+1, j, k)] - vel_field[idx_c]) / dx;
            }
        }
        
        // y-direction advection
        TNum v_face = 0.5 * (v[idx_c] + v[idx(i, j+1 < ny ? j+1 : j, k)]);
        if (has_y_stencil) {
            if (v_face > 0) {
                adv_term += v_face * wenoDerivative(
                    vel_field[idx(i, j-2, k)], 
                    vel_field[idx(i, j-1, k)], 
                    vel_field[idx_c], 
                    vel_field[idx(i, j+1, k)], 
                    vel_field[idx(i, j+2, k)], 
                    dy, true);
            } else {
                adv_term += v_face * wenoDerivative(
                    vel_field[idx(i, j+2, k)], 
                    vel_field[idx(i, j+1, k)], 
                    vel_field[idx_c], 
                    vel_field[idx(i, j-1, k)], 
                    vel_field[idx(i, j-2, k)], 
                    dy, false);
            }
        } else {
            // Fall back to first-order upwind near boundaries
            if (v_face > 0 && j > 0) {
                adv_term += v_face * (vel_field[idx_c] - vel_field[idx(i, j-1, k)]) / dy;
            } else if (v_face < 0 && j < ny-1) {
                adv_term += v_face * (vel_field[idx(i, j+1, k)] - vel_field[idx_c]) / dy;
            }
        }
        
        // z-direction advection
        TNum w_face = 0.5 * (w[idx_c] + w[idx(i, j, k+1 < nz ? k+1 : k)]);
        if (has_z_stencil) {
            if (w_face > 0) {
                adv_term += w_face * wenoDerivative(
                    vel_field[idx(i, j, k-2)], 
                    vel_field[idx(i, j, k-1)], 
                    vel_field[idx_c], 
                    vel_field[idx(i, j, k+1)], 
                    vel_field[idx(i, j, k+2)], 
                    dz, true);
            } else {
                adv_term += w_face * wenoDerivative(
                    vel_field[idx(i, j, k+2)], 
                    vel_field[idx(i, j, k+1)], 
                    vel_field[idx_c], 
                    vel_field[idx(i, j, k-1)], 
                    vel_field[idx(i, j, k-2)], 
                    dz, false);
            }
        } else {
            // Fall back to first-order upwind near boundaries
            if (w_face > 0 && k > 0) {
                adv_term += w_face * (vel_field[idx_c] - vel_field[idx(i, j, k-1)]) / dz;
            } else if (w_face < 0 && k < nz-1) {
                adv_term += w_face * (vel_field[idx(i, j, k+1)] - vel_field[idx_c]) / dz;
            }
        }
        
        return adv_term;
    }
    
    TNum computeDiffusion(const VectorObj<TNum>& vel_field, int i, int j, int k) {
        int idx_c = idx(i, j, k);
        TNum nu_eff = 1.0/Re;
        
        // Add turbulent viscosity if turbulence model is enabled
        if (use_turbulence_model) {
            nu_eff += nu_t[idx_c];
        }
        
        // Second-order central difference for diffusion
        return nu_eff * (
            (vel_field[idx(i+1, j, k)] - 2*vel_field[idx_c] + vel_field[idx(i-1, j, k)]) / (dx*dx) +
            (vel_field[idx(i, j+1, k)] - 2*vel_field[idx_c] + vel_field[idx(i, j-1, k)]) / (dy*dy) +
            (vel_field[idx(i, j, k+1)] - 2*vel_field[idx_c] + vel_field[idx(i, j, k-1)]) / (dz*dz)
        );
    }
    
    void computeIntermediateVelocity() {
        // Store previous velocity
        u_prev = u;
        v_prev = v;
        w_prev = w;
        
        // Compute intermediate velocity field using high-order schemes
        #pragma omp parallel for collapse(3)
        for (int k = 1; k < nz-1; k++) {
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    int idx_c = idx(i, j, k);
                    
                    // Advection terms using WENO scheme
                    TNum u_adv = computeAdvection(u, i, j, k, 0);
                    TNum v_adv = computeAdvection(v, i, j, k, 1);
                    TNum w_adv = computeAdvection(w, i, j, k, 2);
                    
                    // Diffusion terms
                    TNum u_diff = computeDiffusion(u, i, j, k);
                    TNum v_diff = computeDiffusion(v, i, j, k);
                    TNum w_diff = computeDiffusion(w, i, j, k);
                    
                    // Update intermediate velocity with external forces
                    u[idx_c] = u_prev[idx_c] + dt * (-u_adv + u_diff + fx[idx_c]);
                    v[idx_c] = v_prev[idx_c] + dt * (-v_adv + v_diff + fy[idx_c]);
                    w[idx_c] = w_prev[idx_c] + dt * (-w_adv + w_diff + fz[idx_c]);
                }
            }
        }
    }
    
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
        
        p.zero();
        // Solve pressure Poisson equation using iterative solver with improved convergence
        amg.amgVCycle(laplacian, rhs, p, pressure_iterations, 5, 0.25);
    }
    
    void projectVelocity() {
        // Project velocity field to ensure incompressibility
        #pragma omp parallel for collapse(3)
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
    
    void updateTurbulentViscosity() {
        if (!use_turbulence_model) return;
        
        // Smagorinsky-Lilly LES model
        #pragma omp parallel for collapse(3)
        for (int k = 1; k < nz-1; k++) {
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    int idx_c = idx(i, j, k);
                    
                    // Compute strain rate tensor components
                    TNum S11 = (u[idx(i+1, j, k)] - u[idx(i-1, j, k)]) / (2*dx);
                    TNum S22 = (v[idx(i, j+1, k)] - v[idx(i, j-1, k)]) / (2*dy);
                    TNum S33 = (w[idx(i, j, k+1)] - w[idx(i, j, k-1)]) / (2*dz);
                    
                    TNum S12 = 0.5 * ((u[idx(i, j+1, k)] - u[idx(i, j-1, k)]) / (2*dy) + 
                                     (v[idx(i+1, j, k)] - v[idx(i-1, j, k)]) / (2*dx));
                    TNum S13 = 0.5 * ((u[idx(i, j, k+1)] - u[idx(i, j, k-1)]) / (2*dz) + 
                                     (w[idx(i+1, j, k)] - w[idx(i-1, j, k)]) / (2*dx));
                    TNum S23 = 0.5 * ((v[idx(i, j, k+1)] - v[idx(i, j, k-1)]) / (2*dz) + 
                                     (w[idx(i, j+1, k)] - w[idx(i, j-1, k)]) / (2*dy));
                    
                    // Compute magnitude of strain rate tensor
                    TNum S_mag = std::sqrt(2.0 * (S11*S11 + S22*S22 + S33*S33 + 
                                                2.0*(S12*S12 + S13*S13 + S23*S23)));
                    
                    // Compute turbulent viscosity using Smagorinsky model
                    TNum delta = std::cbrt(dx * dy * dz);  // Filter width
                    nu_t[idx_c] = std::pow(Cs * delta, 2) * S_mag;
                }
            }
        }
    }
    
    TNum computeAdaptiveTimeStep() {
        if (!adaptive_dt) return dt;
        
        TNum max_vel = 0.0;
        
        // Find maximum velocity magnitude
        for (int k = 1; k < nz-1; k++) {
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    int idx_c = idx(i, j, k);
                    TNum vel_mag = std::sqrt(u[idx_c]*u[idx_c] + v[idx_c]*v[idx_c] + w[idx_c]*w[idx_c]);
                    max_vel = std::max(max_vel, vel_mag);
                }
            }
        }
        
        // Compute time step based on CFL condition
        TNum dx_min = std::min(std::min(dx, dy), dz);
        TNum dt_cfl = cfl * dx_min / (max_vel > 1e-10 ? max_vel : 1e-10);
        
        // Compute time step based on viscous stability
        TNum nu_max = 1.0/Re;
        if (use_turbulence_model) {
            for (int i = 0; i < nx*ny*nz; i++) {
                nu_max = std::max(nu_max, 1.0/Re + nu_t[i]);
            }
        }
        TNum dt_visc = 0.5 * dx_min * dx_min / nu_max;
        
        // Take minimum of the two constraints
        TNum new_dt = std::min(dt_cfl, dt_visc);
        
        // Limit maximum time step change
        new_dt = std::min(new_dt, 1.2 * dt);
        
        // Enforce maximum time step
        new_dt = std::min(new_dt, dt_max);
        
        return new_dt;
    }

public:
    NavierStokesSolver3D(int nx_, int ny_, int nz_, TNum dx_, TNum dy_, TNum dz_, TNum dt_, TNum Re_)
        : nx(nx_), ny(ny_), nz(nz_), dx(dx_), dy(dy_), dz(dz_), dt(dt_), dt_max(dt_ * 5.0), Re(Re_),
          cfl(0.5), adaptive_dt(false), use_turbulence_model(false), Cs(0.17), pressure_iterations(3),
          pressure_tolerance(1e-6),
          u(nx*ny*nz, 0.0), v(nx*ny*nz, 0.0), w(nx*ny*nz, 0.0), p(nx*ny*nz, 0.0),
          u_prev(nx*ny*nz, 0.0), v_prev(nx*ny*nz, 0.0), w_prev(nx*ny*nz, 0.0),
          fx(nx*ny*nz, 0.0), fy(nx*ny*nz, 0.0), fz(nx*ny*nz, 0.0),
          nu_t(nx*ny*nz, 0.0) {
        
        // Default boundary conditions (zero velocity)
        u_bc = [](TNum x, TNum y, TNum z, TNum t) { return 0.0; };
        v_bc = [](TNum x, TNum y, TNum z, TNum t) { return 0.0; };
        w_bc = [](TNum x, TNum y, TNum z, TNum t) { return 0.0; };
        
        // Build Laplacian matrix for pressure Poisson equation
        buildLaplacianMatrix();
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
    
    // Set initial conditions
    void setInitialConditions(
        std::function<TNum(TNum, TNum, TNum)> u_init,
        std::function<TNum(TNum, TNum, TNum)> v_init,
        std::function<TNum(TNum, TNum, TNum)> w_init) {
        
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    int index = idx(i, j, k);
                    u[index] = u_init(i*dx, j*dy, k*dz);
                    v[index] = v_init(i*dx, j*dy, k*dz);
                    w[index] = w_init(i*dx, j*dy, k*dz);
                }
            }
        }
        
        u_prev = u;
        v_prev = v;
        w_prev = w;
    }
    
    // Advance simulation by one time step
    void step(TNum time) {
        // Update turbulent viscosity if LES model is enabled
        if (use_turbulence_model) {
            updateTurbulentViscosity();
        }
        
        // Update time step if adaptive time stepping is enabled
        if (adaptive_dt) {
            dt = computeAdaptiveTimeStep();
        }
        
        // Apply boundary conditions
        applyBoundaryConditions(time);
        
        // Compute intermediate velocity field
        computeIntermediateVelocity();
        
        // Solve pressure Poisson equation
        solvePressurePoisson();
        
        // Project velocity field to ensure incompressibility
        projectVelocity();
        
        // Apply boundary conditions again to ensure they are enforced
        applyBoundaryConditions(time);
    }
    
    // Run simulation for specified duration
    void simulate(TNum duration, int output_freq = 1) {
        int num_steps = static_cast<int>(duration / dt);
        
        for (int step = 0; step < num_steps; step++) {
            TNum current_time = step * dt;
            
            // Advance simulation by one time step
            step(current_time);
            
            // Output progress
            if (step % output_freq == 0) {
                std::cout << "Step " << step << "/" << num_steps 
                          << ", Time: " << current_time 
                          << ", Kinetic Energy: " << calculateKineticEnergy() << std::endl;
            }
        }
    }
    
    // Calculate kinetic energy of the flow
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
        
        return energy / ((nx-2) * (ny-2) * (nz-2));
    }
    
    // Calculate vorticity field
    void calculateVorticity(VectorObj<TNum>& vort_x, VectorObj<TNum>& vort_y, VectorObj<TNum>& vort_z) const {
        // Resize vorticity vectors if needed
        if (vort_x.size() != nx*ny*nz) vort_x.resize(nx*ny*nz, 0.0);
        if (vort_y.size() != nx*ny*nz) vort_y.resize(nx*ny*nz, 0.0);
        if (vort_z.size() != nx*ny*nz) vort_z.resize(nx*ny*nz, 0.0);
        
        // Calculate vorticity components using central differences
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
        
        // Set vorticity to zero at boundaries
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                vort_x[idx(0, j, k)] = vort_y[idx(0, j, k)] = vort_z[idx(0, j, k)] = 0.0;
                vort_x[idx(nx-1, j, k)] = vort_y[idx(nx-1, j, k)] = vort_z[idx(nx-1, j, k)] = 0.0;
            }
        }
        
        for (int k = 0; k < nz; k++) {
            for (int i = 0; i < nx; i++) {
                vort_x[idx(i, 0, k)] = vort_y[idx(i, 0, k)] = vort_z[idx(i, 0, k)] = 0.0;
                vort_x[idx(i, ny-1, k)] = vort_y[idx(i, ny-1, k)] = vort_z[idx(i, ny-1, k)] = 0.0;
            }
        }
        
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                vort_x[idx(i, j, 0)] = vort_y[idx(i, j, 0)] = vort_z[idx(i, j, 0)] = 0.0;
                vort_x[idx(i, j, nz-1)] = vort_y[idx(i, j, nz-1)] = vort_z[idx(i, j, nz-1)] = 0.0;
            }
        }
    }
    
    // Calculate divergence of velocity field (should be close to zero)
    VectorObj<TNum> calculateDivergence() const {
        VectorObj<TNum> div(nx*ny*nz, 0.0);
        
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
    
    // Add external force field
    void addForceField(
        std::function<TNum(TNum, TNum, TNum, TNum)> fx_func,
        std::function<TNum(TNum, TNum, TNum, TNum)> fy_func,
        std::function<TNum(TNum, TNum, TNum, TNum)> fz_func,
        TNum time) {
        
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    int idx_c = idx(i, j, k);
                    fx[idx_c] = fx_func(i*dx, j*dy, k*dz, time);
                    fy[idx_c] = fy_func(i*dx, j*dy, k*dz, time);
                    fz[idx_c] = fz_func(i*dx, j*dy, k*dz, time);
                }
            }
        }
    }
    
    // Enable/disable adaptive time stepping
    void setAdaptiveTimeStep(bool enable, TNum cfl_number = 0.5) {
        adaptive_dt = enable;
        cfl = cfl_number;
    }
    
    // Enable/disable turbulence modeling
    void enableTurbulenceModel(bool enable, TNum Cs_value = 0.17) {
        use_turbulence_model = enable;
        Cs = Cs_value;
    }
    
    // Set pressure solver parameters
    void setPressureSolverParams(int iterations, TNum tolerance) {
        pressure_iterations = iterations;
        pressure_tolerance = tolerance;
    }
    
    // Accessor methods
    int getNx() const { return nx; }
    int getNy() const { return ny; }
    int getNz() const { return nz; }
    TNum getDx() const { return dx; }
    TNum getDy() const { return dy; }
    TNum getDz() const { return dz; }
    TNum getDt() const { return dt; }
    TNum getRe() const { return Re; }
    
    // Get velocity and pressure values at specific grid points
    TNum getU(int i, int j, int k) const { return u[idx(i, j, k)]; }
    TNum getV(int i, int j, int k) const { return v[idx(i, j, k)]; }
    TNum getW(int i, int j, int k) const { return w[idx(i, j, k)]; }
    TNum getP(int i, int j, int k) const { return p[idx(i, j, k)]; }
    
    // Get velocity and pressure fields
    const VectorObj<TNum>& getUField() const { return u; }
    const VectorObj<TNum>& getVField() const { return v; }
    const VectorObj<TNum>& getWField() const { return w; }
    const VectorObj<TNum>& getPField() const { return p; }
};

#endif // NAVIER_STOKES_HPP
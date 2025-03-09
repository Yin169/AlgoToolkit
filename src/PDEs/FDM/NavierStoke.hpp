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

template <typename TNum>
class NavierStokesSolver3D {
private:
    // Grid dimensions
    int nx, ny, nz;
    TNum dx, dy, dz;
    TNum dt;
    TNum Re;  // Reynolds number
    
    // Velocity components (u, v, w) and pressure (p)
    VectorObj<TNum> u, v, w, p;
    VectorObj<TNum> u_prev, v_prev, w_prev;
    
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
    
    void computeIntermediateVelocity() {
        // Store previous velocity
        u_prev = u;
        v_prev = v;
        w_prev = w;
        
        // Compute intermediate velocity field using explicit scheme
        for (int k = 1; k < nz-1; k++) {
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    int idx_c = idx(i, j, k);
                    
                    // Advection terms (upwind scheme)
                    TNum u_adv = 0, v_adv = 0, w_adv = 0;
                    
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
                    
                    // Diffusion terms (central difference)
                    TNum u_diff = (u[idx(i+1, j, k)] - 2*u[idx_c] + u[idx(i-1, j, k)]) / (dx*dx) +
                                 (u[idx(i, j+1, k)] - 2*u[idx_c] + u[idx(i, j-1, k)]) / (dy*dy) +
                                 (u[idx(i, j, k+1)] - 2*u[idx_c] + u[idx(i, j, k-1)]) / (dz*dz);
                    
                    TNum v_diff = (v[idx(i+1, j, k)] - 2*v[idx_c] + v[idx(i-1, j, k)]) / (dx*dx) +
                                 (v[idx(i, j+1, k)] - 2*v[idx_c] + v[idx(i, j-1, k)]) / (dy*dy) +
                                 (v[idx(i, j, k+1)] - 2*v[idx_c] + v[idx(i, j, k-1)]) / (dz*dz);
                    
                    TNum w_diff = (w[idx(i+1, j, k)] - 2*w[idx_c] + w[idx(i-1, j, k)]) / (dx*dx) +
                                 (w[idx(i, j+1, k)] - 2*w[idx_c] + w[idx(i, j-1, k)]) / (dy*dy) +
                                 (w[idx(i, j, k+1)] - 2*w[idx_c] + w[idx(i, j, k-1)]) / (dz*dz);
                    
                    // Update intermediate velocity
                    u[idx_c] = u_prev[idx_c] + dt * (-u_adv + (1.0/Re) * u_diff);
                    v[idx_c] = v_prev[idx_c] + dt * (-v_adv + (1.0/Re) * v_diff);
                    w[idx_c] = w_prev[idx_c] + dt * (-w_adv + (1.0/Re) * w_diff);
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
        // Solve pressure Poisson equation using iterative solver
        amg.amgVCycle(laplacian, rhs, p, 3, 5, 0.25);
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

public:
    NavierStokesSolver3D(int nx_, int ny_, int nz_, TNum dx_, TNum dy_, TNum dz_, TNum dt_, TNum Re_)
        : nx(nx_), ny(ny_), nz(nz_), dx(dx_), dy(dy_), dz(dz_), dt(dt_), Re(Re_),
          u(nx*ny*nz, 0.0), v(nx*ny*nz, 0.0), w(nx*ny*nz, 0.0), p(nx*ny*nz, 0.0),
          u_prev(nx*ny*nz, 0.0), v_prev(nx*ny*nz, 0.0), w_prev(nx*ny*nz, 0.0) {
        
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
        // Apply boundary conditions
        applyBoundaryConditions(time);
        
        // Compute intermediate velocity field
        computeIntermediateVelocity();
        
        // Solve pressure Poisson equation
        solvePressurePoisson();
        
        // Project velocity field to ensure incompressibility
        projectVelocity();
    }
    
    // Run simulation for specified duration
    void simulate(TNum duration, int output_freq = 1) {
        int num_steps = static_cast<int>(duration / dt);
        
        for (int step = 0; step < num_steps; step++) {
            TNum current_time = step * dt;
            this->step(current_time);
            
            if (step % output_freq == 0) {
                std::cout << "Time: " << current_time << ", Step: " << step << std::endl;
            }
        }
    }
    
    // Get velocity and pressure fields
    const VectorObj<TNum>& getU() const { return u; }
    const VectorObj<TNum>& getV() const { return v; }
    const VectorObj<TNum>& getW() const { return w; }
    const VectorObj<TNum>& getP() const { return p; }
    
    // Get velocity at specific grid point
    TNum getU(int i, int j, int k) const { return u[idx(i, j, k)]; }
    TNum getV(int i, int j, int k) const { return v[idx(i, j, k)]; }
    TNum getW(int i, int j, int k) const { return w[idx(i, j, k)]; }
    TNum getP(int i, int j, int k) const { return p[idx(i, j, k)]; }
    
    // Get grid dimensions
    int getNx() const { return nx; }
    int getNy() const { return ny; }
    int getNz() const { return nz; }
    TNum getDx() const { return dx; }
    TNum getDy() const { return dy; }
    TNum getDz() const { return dz; }
	TNum getDt() const { return dt; }
	TNum getRe() const { return Re; }
	
    
    // Calculate vorticity (curl of velocity field)
    void calculateVorticity(VectorObj<TNum>& vort_x, VectorObj<TNum>& vort_y, VectorObj<TNum>& vort_z) const {
        vort_x.resize(nx * ny * nz);
        vort_y.resize(nx * ny * nz);
        vort_z.resize(nx * ny * nz);
        
        for (int k = 1; k < nz-1; k++) {
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    int index = idx(i, j, k);
                    
                    // ω_x = ∂w/∂y - ∂v/∂z
                    vort_x[index] = (w[idx(i, j+1, k)] - w[idx(i, j-1, k)]) / (2*dy) - 
                                   (v[idx(i, j, k+1)] - v[idx(i, j, k-1)]) / (2*dz);
                    
                    // ω_y = ∂u/∂z - ∂w/∂x
                    vort_y[index] = (u[idx(i, j, k+1)] - u[idx(i, j, k-1)]) / (2*dz) - 
                                   (w[idx(i+1, j, k)] - w[idx(i-1, j, k)]) / (2*dx);
                    
                    // ω_z = ∂v/∂x - ∂u/∂y
                    vort_z[index] = (v[idx(i+1, j, k)] - v[idx(i-1, j, k)]) / (2*dx) - 
                                   (u[idx(i, j+1, k)] - u[idx(i, j-1, k)]) / (2*dy);
                }
            }
        }
    }
    
    // Calculate kinetic energy
    TNum calculateKineticEnergy() const {
        TNum energy = 0.0;
        
        for (int k = 1; k < nz-1; k++) {
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    int index = idx(i, j, k);
                    energy += 0.5 * (u[index]*u[index] + v[index]*v[index] + w[index]*w[index]);
                }
            }
        }
        
        return energy * dx * dy * dz;
    }
    
    // Calculate divergence of velocity field (should be close to zero for incompressible flow)
    VectorObj<TNum> calculateDivergence() const {
        VectorObj<TNum> div(nx * ny * nz, 0.0);
        
        for (int k = 1; k < nz-1; k++) {
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    int index = idx(i, j, k);
                    
                    div[index] = (u[idx(i+1, j, k)] - u[idx(i-1, j, k)]) / (2*dx) +
                                (v[idx(i, j+1, k)] - v[idx(i, j-1, k)]) / (2*dy) +
                                (w[idx(i, j, k+1)] - w[idx(i, j, k-1)]) / (2*dz);
                }
            }
        }
        
        return div;
    }
};

#endif // NAVIER_STOKES_HPP

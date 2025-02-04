#ifndef VORTICITYSTREAMSOLVER_HPP
#define VORTICITYSTREAMSOLVER_HPP

#include "../../src/Obj/SparseObj.hpp"
#include "../../src/Obj/VectorObj.hpp"
#include "../../src/LinearAlgebra/Preconditioner/MultiGrid.hpp"
#include "../../src/LinearAlgebra/Krylov/GMRES.hpp"
#include <cmath>
#include <functional>
#include <stdexcept>
#include <iostream>

template<typename TNum>
class VorticityStreamSolver {
private:
    // Configuration parameters
    int nx, ny;
    TNum dx, dy;
    TNum Re;
    TNum U;
    TNum tolerance;
    TNum cfl_limit;

    // Field variables
    SparseMatrixCSC<TNum> laplacian;
    VectorObj<TNum> vorticity;
    VectorObj<TNum> streamFunction;
    VectorObj<TNum> u_velocity;
    VectorObj<TNum> v_velocity;

    // Solver components
    AlgebraicMultiGrid<TNum, VectorObj<TNum>> multigrid;
    int mg_levels = 2;  // Further reduced for better stability
    int mg_smoothing_steps = 40;  // Increased for better smoothing
    TNum mg_theta = 0.3;  // Increased for stronger coarsening
    int mg_max_cycles = 150;  // Increased to ensure convergence
    TNum mg_tolerance = 1e-5;  // Further relaxed for better convergence

    void buildLaplacianMatrix() {
        const int n = nx * ny;
        laplacian = SparseMatrixCSC<TNum>(n, n);
        
        const TNum dx2i = 1.0 / (dx * dx);
        const TNum dy2i = 1.0 / (dy * dy);
        
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int idx = i + j * nx;
                
                if (i == 0 || i == nx-1 || j == 0 || j == ny-1) {
                    laplacian.addValue(idx, idx, 1.0);
                } else {
                    const TNum scale = -2.0 * (dx2i + dy2i);
                    laplacian.addValue(idx, idx, scale);
                    laplacian.addValue(idx, idx-1, dx2i);
                    laplacian.addValue(idx, idx+1, dx2i);
                    laplacian.addValue(idx, idx-nx, dy2i);
                    laplacian.addValue(idx, idx+nx, dy2i);
                }
            }
        }
        laplacian.finalize();
    }

    void computeVelocityField() {
        u_velocity.zero();
        v_velocity.zero();

        for (int j = 1; j < ny-1; ++j) {
            for (int i = 1; i < nx-1; ++i) {
                const int idx = i + j * nx;
                u_velocity[idx] = (streamFunction[idx+nx] - streamFunction[idx-nx])/(2.0*dy);
                v_velocity[idx] = -(streamFunction[idx+1] - streamFunction[idx-1])/(2.0*dx);
            }
        }

        // Top wall (moving lid)
        for (int i = 0; i < nx; ++i) {
            u_velocity[i + (ny-1)*nx] = U;
        }
    }

    void updateBoundaryConditions() {
        // Stream function boundary conditions (using second-order accurate discretization)
        for (int i = 0; i < nx; ++i) {
            streamFunction[i] = 0.0;
            streamFunction[i + (ny-1)*nx] = 0.0;
        }
        for (int j = 0; j < ny; ++j) {
            streamFunction[j*nx] = 0.0;
            streamFunction[(j+1)*nx-1] = 0.0;
        }

        const TNum dy2i = 1.0/(dy*dy);
        const TNum dx2i = 1.0/(dx*dx);

        // Top wall (moving lid) - second order accurate
        for (int i = 1; i < nx-1; ++i) {
            const int top_idx = i + (ny-1)*nx;
            vorticity[top_idx] = (-3.0*streamFunction[top_idx] + 4.0*streamFunction[top_idx-nx] - streamFunction[top_idx-2*nx])/(2.0*dy*dy) - 3.0*U/(2.0*dy);
        }
        
        // Bottom wall - second order accurate
        for (int i = 1; i < nx-1; ++i) {
            vorticity[i] = (-3.0*streamFunction[i] + 4.0*streamFunction[i+nx] - streamFunction[i+2*nx])/(2.0*dy*dy);
        }
        
        // Vertical walls - second order accurate
        for (int j = 1; j < ny-1; ++j) {
            const int left_idx = j*nx;
            const int right_idx = (j+1)*nx-1;
            vorticity[left_idx] = (-3.0*streamFunction[left_idx] + 4.0*streamFunction[left_idx+1] - streamFunction[left_idx+2])/(2.0*dx*dx);
            vorticity[right_idx] = (-3.0*streamFunction[right_idx] + 4.0*streamFunction[right_idx-1] - streamFunction[right_idx-2])/(2.0*dx*dx);
        }

        // Corner vorticity - using one-sided differences
        vorticity[0] = (-3.0*streamFunction[0] + 4.0*streamFunction[1] - streamFunction[2])/(2.0*dx*dx);
        vorticity[nx-1] = (-3.0*streamFunction[nx-1] + 4.0*streamFunction[nx-2] - streamFunction[nx-3])/(2.0*dx*dx);
        vorticity[(ny-1)*nx] = (-3.0*streamFunction[(ny-1)*nx] + 4.0*streamFunction[(ny-2)*nx] - streamFunction[(ny-3)*nx])/(2.0*dy*dy);
        vorticity[ny*nx-1] = (-3.0*streamFunction[ny*nx-1] + 4.0*streamFunction[(ny-1)*nx-1] - streamFunction[(ny-2)*nx-1])/(2.0*dy*dy);
    }

    VectorObj<TNum> computeVorticityDerivative(const VectorObj<TNum>& w) const {
        VectorObj<TNum> dwdt(w.size(), 0.0);
        const TNum re_inv = 1.0/Re;
        const TNum dx2i = 1.0/(dx*dx);
        const TNum dy2i = 1.0/(dy*dy);
        const TNum dx2 = 2.0*dx;
        const TNum dy2 = 2.0*dy;

        for (int j = 1; j < ny-1; ++j) {
            for (int i = 1; i < nx-1; ++i) {
                const int idx = i + j*nx;
                
                const TNum dwdx = (w[idx+1] - w[idx-1])/dx2;
                const TNum dwdy = (w[idx+nx] - w[idx-nx])/dy2;
                
                const TNum d2wdx2 = (w[idx+1] - 2.0*w[idx] + w[idx-1])*dx2i;
                const TNum d2wdy2 = (w[idx+nx] - 2.0*w[idx] + w[idx-nx])*dy2i;
                
                const TNum conv = u_velocity[idx]*dwdx + v_velocity[idx]*dwdy;
                const TNum diff = d2wdx2 + d2wdy2;
                
                dwdt[idx] = -conv + re_inv*diff;
            }
        }
        return dwdt;
    }

    TNum computeCFL(TNum dt) const {
        TNum max_vel = 0.0;
        for (int idx = 0; idx < nx*ny; ++idx) {
            const TNum vel_mag = std::hypot(u_velocity[idx], v_velocity[idx]);
            max_vel = std::max(max_vel, vel_mag);
        }
        return dt * max_vel * std::max(1.0/dx, 1.0/dy);
    }

    bool checkConvergence(const VectorObj<TNum>& prev) const {
        TNum max_err = 0.0;
        TNum max_val = 1e-10;
        TNum l2_err = 0.0;
        TNum l2_val = 0.0;
        
        for (size_t i = 0; i < vorticity.size(); ++i) {
            const TNum diff = std::abs(vorticity[i] - prev[i]);
            const TNum val = std::abs(vorticity[i]);
            max_err = std::max(max_err, diff);
            max_val = std::max(max_val, val);
            l2_err += diff * diff;
            l2_val += val * val;
        }
        
        // Use both max norm and L2 norm for convergence check
        return (max_err/max_val < tolerance) && (std::sqrt(l2_err/l2_val) < tolerance);
    }

public:
    VorticityStreamSolver(int nx_, int ny_, TNum Re_, TNum U_ = 1.0,
                         TNum tol = 1e-6, TNum cfl = 0.5)
        : nx(nx_), ny(ny_), Re(Re_), U(U_), tolerance(tol), cfl_limit(cfl) {

        if (nx < 3 || ny < 3) throw std::invalid_argument("Grid size must be at least 3x3");
        if (Re <= 0) throw std::invalid_argument("Reynolds number must be positive");

        dx = 1.0/(nx-1);
        dy = 1.0/(ny-1);

        const int n = nx * ny;
        vorticity = VectorObj<TNum>(n, 0.0);
        streamFunction = VectorObj<TNum>(n, 0.0);
        u_velocity = VectorObj<TNum>(n, 0.0);
        v_velocity = VectorObj<TNum>(n, 0.0);

        buildLaplacianMatrix();
        updateBoundaryConditions();
    }

    bool solve(TNum dt, int max_steps) {
        VectorObj<TNum> poisson_rhs(nx*ny);
        VectorObj<TNum> residual(nx*ny);
        VectorObj<TNum> dwdt(nx*ny);
        const TNum under_relax = 0.5;  // Reduced under-relaxation factor for better stability

        for (int step = 0; step < max_steps; ++step) {
            poisson_rhs = vorticity * -1.0;
            
            TNum initial_res_norm = 0.0;
            residual = poisson_rhs - (laplacian * streamFunction);
            for (size_t i = 0; i < residual.size(); ++i) {
                initial_res_norm = std::max(initial_res_norm, std::abs(residual[i]));
            }

            bool poisson_converged = false;
            for (int cycle = 0; cycle < mg_max_cycles; ++cycle) {
                multigrid.amgVCycle(laplacian, poisson_rhs, streamFunction, 
                                  mg_levels, mg_smoothing_steps, mg_theta);
                
                residual = poisson_rhs - (laplacian * streamFunction);
                TNum res_norm = 0.0;
                for (size_t i = 0; i < residual.size(); ++i) {
                    res_norm = std::max(res_norm, std::abs(residual[i]));
                }
                
                if (res_norm < mg_tolerance * initial_res_norm) {
                    poisson_converged = true;
                    break;
                }
            }

            if (!poisson_converged) {
                throw std::runtime_error("Poisson solver did not converge");
            }

            computeVelocityField();
            updateBoundaryConditions();

            const TNum cfl = computeCFL(dt);
            if (cfl > cfl_limit) {
                dt *= 0.25 * cfl_limit / cfl;  // More aggressive time step reduction
                continue;
            }

            const VectorObj<TNum> old_vorticity = vorticity;
            dwdt = computeVorticityDerivative(vorticity);
            
            // Apply under-relaxation in time integration
            for (size_t i = 0; i < vorticity.size(); ++i) {
                vorticity[i] = old_vorticity[i] + under_relax * dt * dwdt[i];
            }
            
            updateBoundaryConditions();

            if (step % 100 == 0) {
                if (checkConvergence(old_vorticity)) {
                    std::cout << "Converged after " << step << " steps\n";
                    return true;
                }
                std::cout << "Step " << step << ", CFL = " << cfl 
                         << ", dt = " << dt << std::endl;
            }
        }
        return false;
    }

    // Accessors
    const VectorObj<TNum>& getVorticity() const noexcept { return vorticity; }
    const VectorObj<TNum>& getStreamFunction() const noexcept { return streamFunction; }
    const VectorObj<TNum>& getUVelocity() const noexcept { return u_velocity; }
    const VectorObj<TNum>& getVVelocity() const noexcept { return v_velocity; }
    TNum getDx() const noexcept { return dx; }
    TNum getDy() const noexcept { return dy; }
    int getNx() const noexcept { return nx; }
    int getNy() const noexcept { return ny; }
};

#endif // VORTICITYSTREAMSOLVER_HPP
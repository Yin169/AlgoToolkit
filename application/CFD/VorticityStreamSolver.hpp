#ifndef VORTICITYSTREAMSOLVER_HPP
#define VORTICITYSTREAMSOLVER_HPP

#include "../../src/Obj/SparseObj.hpp" // Assuming these paths are correct relative to your project
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
    SparseMatrixCSC<TNum> laplacian; // For Poisson equation of stream function
    SparseMatrixCSC<TNum> diffusion_matrix; // For implicit diffusion term in vorticity equation
    VectorObj<TNum> vorticity;
    VectorObj<TNum> streamFunction;
    VectorObj<TNum> u_velocity;
    VectorObj<TNum> v_velocity;

    // Solver components
    AlgebraicMultiGrid<TNum, VectorObj<TNum>> multigrid;
    int mg_levels = 7;
    int mg_smoothing_steps = 1000;
    TNum mg_theta = 0.3;
    int mg_max_cycles = 150;
    TNum mg_tolerance = 1e-5;

    void buildLaplacianMatrix() {
        const int n = nx * ny;
        laplacian = SparseMatrixCSC<TNum>(n, n);
        diffusion_matrix = SparseMatrixCSC<TNum>(n, n); // Initialize diffusion matrix

        const TNum dx2i = 1.0 / (dx * dx);
        const TNum dy2i = 1.0 / (dy * dy);

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int idx = i + j * nx;

                if (i == 0 || i == nx-1 || j == 0 || j == ny-1) {
                    laplacian.addValue(idx, idx, 1.0);
                    diffusion_matrix.addValue(idx, idx, 1.0); // Boundary conditions for implicit diffusion
                } else {
                    const TNum laplacian_scale = -2.0 * (dx2i + dy2i);
                    laplacian.addValue(idx, idx, laplacian_scale);
                    laplacian.addValue(idx, idx-1, dx2i);
                    laplacian.addValue(idx, idx+1, dx2i);
                    laplacian.addValue(idx, idx-nx, dy2i);
                    laplacian.addValue(idx, idx+nx, dy2i);

                    // For implicit diffusion: I - 0.5*dt*Re*Laplacian 
                    const TNum diff_scale_diag = 1.0 - 0.5 * Re * (-laplacian_scale); // Modified diagonal
                    const TNum diff_scale_off_diag = -0.5 * Re * (dx2i);             // Modified off-diagonals (x direction)
                    const TNum diff_scale_off_diag_y = -0.5 * Re * (dy2i);           // Modified off-diagonals (y direction)

                    diffusion_matrix.addValue(idx, idx, diff_scale_diag);
                    diffusion_matrix.addValue(idx, idx-1, diff_scale_off_diag);
                    diffusion_matrix.addValue(idx, idx+1, diff_scale_off_diag);
                    diffusion_matrix.addValue(idx, idx-nx, diff_scale_off_diag_y);
                    diffusion_matrix.addValue(idx, idx+nx, diff_scale_off_diag_y);
                }
            }
        }
        laplacian.finalize();
        diffusion_matrix.finalize();
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
        // Update stream function BCs
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

        // Top wall (moving lid) - second order accurate for vorticity
        for (int i = 1; i < nx-1; ++i) {
            const int top_idx = i + (ny-1)*nx;
            vorticity[top_idx] = (-3.0*streamFunction[top_idx] + 4.0*streamFunction[top_idx-nx] - streamFunction[top_idx-2*nx])/(2.0*dy*dy) - 3.0*U/(2.0*dy);
        }

        // Bottom wall - second order accurate for vorticity
        for (int i = 1; i < nx-1; ++i) {
            vorticity[i] = (-3.0*streamFunction[i] + 4.0*streamFunction[i+nx] - streamFunction[i+2*nx])/(2.0*dy*dy);
        }

        // Vertical walls - second order accurate for vorticity
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

    VectorObj<TNum> computeConvectionTerm(const VectorObj<TNum>& w) const {
        VectorObj<TNum> convection(w.size(), 0.0);
        const TNum dx2 = 2.0*dx;
        const TNum dy2 = 2.0*dy;

        for (int j = 1; j < ny-1; ++j) {
            for (int i = 1; i < nx-1; ++i) {
                const int idx = i + j*nx;
                const TNum dwdx = (w[idx+1] - w[idx-1]) / dx2;
                const TNum dwdy = (w[idx+nx] - w[idx-nx]) / dy2;
                convection[idx] = u_velocity[idx]*dwdx + v_velocity[idx]*dwdy;
            }
        }
        return convection;
    }

    VectorObj<TNum> computeDiffusionTerm(const VectorObj<TNum>& w) const {
        VectorObj<TNum> diffusion(w.size(), 0.0);
        const TNum dx2i = 1.0/(dx*dx);
        const TNum dy2i = 1.0/(dy*dy);

        for (int j = 1; j < ny-1; ++j) {
            for (int i = 1; i < nx-1; ++i) {
                const int idx = i + j*nx;
                const TNum d2wdx2 = (w[idx+1] - 2.0*w[idx] + w[idx-1]) * dx2i;
                const TNum d2wdy2 = (w[idx+nx] - 2.0*w[idx] + w[idx-nx]) * dy2i;
                diffusion[idx] = d2wdx2 + d2wdy2;
            }
        }
        return diffusion;
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
        return (max_err / max_val < tolerance) && (std::sqrt(l2_err/l2_val) < tolerance);
    }

public:
    VorticityStreamSolver(int nx_, int ny_, TNum Re_, TNum U_ = 1.0,
                            TNum tol = 1e-6, TNum cfl = 0.4)
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

        buildLaplacianMatrix(); // Builds both laplacian and diffusion_matrix
        updateBoundaryConditions();
    }

    bool solve(TNum dt, int max_steps) {
        VectorObj<TNum> poisson_rhs(nx*ny);
        VectorObj<TNum> residual(nx*ny);
        VectorObj<TNum> convection_term_vec(nx*ny);
        VectorObj<TNum> rhs_vorticity(nx*ny);

        for (int step = 0; step < max_steps; ++step) {
            // --- Poisson Solver for streamFunction ---
            // Update rhs based on current vorticity
            poisson_rhs = vorticity * -1.0;

            residual = poisson_rhs - (laplacian * streamFunction);
            TNum initial_res_norm = 0.0;
            for (size_t i = 0; i < residual.size(); ++i) {
                initial_res_norm = std::max(initial_res_norm, std::abs(residual[i]));
            }

            bool poisson_converged = false;
            int poisson_cycles = 0;
            for (int cycle = 0; cycle < mg_max_cycles; ++cycle) {
                multigrid.amgVCycle(laplacian, poisson_rhs, streamFunction,
                                    mg_levels, mg_smoothing_steps, mg_theta);
                poisson_cycles++;

                residual = poisson_rhs - (laplacian * streamFunction);
                TNum res_norm = 0.0;
                for (size_t i = 0; i < residual.size(); ++i) {
                    res_norm = std::max(res_norm, std::abs(residual[i]));
                }
                if (res_norm / initial_res_norm < mg_tolerance) {
                    poisson_converged = true;
                    break;
                }
            }
            if (!poisson_converged) {
                throw std::runtime_error("Poisson solver did not converge after " +
                                            std::to_string(poisson_cycles) + " cycles");
            }

            // Update velocity and boundary conditions after solving stream function
            computeVelocityField();
            updateBoundaryConditions();

            // --- Time step modification based on CFL condition ---
            const TNum cfl = computeCFL(dt);
            if (cfl > cfl_limit) {
                dt *= 0.5 * cfl_limit / cfl;
                continue;
            }

            // --- Convection update (explicit) ---
            const VectorObj<TNum> old_vorticity = vorticity;
            convection_term_vec = computeConvectionTerm(vorticity);
            rhs_vorticity = old_vorticity + convection_term_vec * dt;

            // --- Diffusion Solver for vorticity ---
            int diffusion_mg_max_cycles = 200; // Increased cycles for diffusion solver
            int diffusion_mg_smoothing_steps = 100; // Increased smoothing steps
            residual.zero();
            residual = rhs_vorticity - (diffusion_matrix * vorticity);
            TNum initial_diff_res_norm = 0.0;
            for (size_t i = 0; i < residual.size(); ++i) {
                initial_diff_res_norm = std::max(initial_diff_res_norm, std::abs(residual[i]));
            }
            bool diffusion_converged = false;
            int diffusion_cycles = 0;
            for (int cycle = 0; cycle < diffusion_mg_max_cycles; ++cycle) {
                multigrid.amgVCycle(diffusion_matrix, rhs_vorticity, vorticity,
                                    mg_levels, diffusion_mg_smoothing_steps, mg_theta);
                diffusion_cycles++;

                residual = rhs_vorticity - (diffusion_matrix * vorticity);
                TNum res_norm = 0.0;
                for (size_t i = 0; i < residual.size(); ++i) {
                    res_norm = std::max(res_norm, std::abs(residual[i]));
                }
                if (res_norm / initial_diff_res_norm < mg_tolerance) {
                    diffusion_converged = true;
                    break;
                }
            }
            if (!diffusion_converged) {
                throw std::runtime_error("Diffusion solver did not converge after " +
                                            std::to_string(diffusion_cycles) + " cycles");
            }

            // Update boundary conditions after diffusion
            updateBoundaryConditions();

            // --- Convergence check ---
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
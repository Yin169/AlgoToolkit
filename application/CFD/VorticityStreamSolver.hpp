#ifndef VORTICITYSTREAMSOLVER_HPP
#define VORTICITYSTREAMSOLVER_HPP

#include "../../src/Obj/SparseObj.hpp"
#include "../../src/Obj/VectorObj.hpp"
#include "../../src/LinearAlgebra/Preconditioner/MultiGrid.hpp"
#include "../../src/LinearAlgebra/Krylov/GMRES.hpp"
#include "../../src/ODE/RungeKutta.hpp"
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
    GMRES<TNum> gmres;
    int gmres_max_iter = 1000;
    int gmres_krylov_dim = 30;
    TNum gmres_tol = 1e-6;

    void buildLaplacianMatrix() {
        const int n = nx * ny;
        laplacian = SparseMatrixCSC<TNum>(n, n);
        
        const TNum dx2i = 1.0 / (dx * dx);
        const TNum dy2i = 1.0 / (dy * dy);
        
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int idx = i + j * nx;
                
                if (i == 0 || i == nx-1 || j == 0 || j == ny-1) {
                    // Boundary identity equations
                    laplacian.addValue(idx, idx, 1.0);
                } else {
                    // Interior Laplacian stencil
                    laplacian.addValue(idx, idx, -2.0*(dx2i + dy2i));
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
        // Interior points using central differences
        for (int j = 1; j < ny-1; ++j) {
            for (int i = 1; i < nx-1; ++i) {
                const int idx = i + j * nx;
                u_velocity[idx] = (streamFunction[idx+nx] - streamFunction[idx-nx])/(2.0*dy);
                v_velocity[idx] = -(streamFunction[idx+1] - streamFunction[idx-1])/(2.0*dx);
            }
        }

        // Boundary conditions
        // Top wall (moving lid)
        for (int i = 0; i < nx; ++i) {
            const int top_idx = i + (ny-1)*nx;
            u_velocity[top_idx] = U;
            v_velocity[top_idx] = 0.0;
        }
        // Bottom wall
        u_velocity.zero();
        v_velocity.zero();
        // Vertical walls
        for (int j = 0; j < ny; ++j) {
            u_velocity[j*nx] = u_velocity[(j+1)*nx-1] = 0.0;
            v_velocity[j*nx] = v_velocity[(j+1)*nx-1] = 0.0;
        }
    }

    void updateBoundaryConditions() {
        // Stream function boundary conditions (Dirichlet zero)
        for (int i = 0; i < nx; ++i) {
            streamFunction[i] = 0.0;                   // Bottom
            streamFunction[i + (ny-1)*nx] = 0.0;       // Top
        }
        for (int j = 0; j < ny; ++j) {
            streamFunction[j*nx] = 0.0;                // Left
            streamFunction[(j+1)*nx-1] = 0.0;          // Right
        }

        // Vorticity boundary conditions (Neumann-type for streamfunction)
        // Top wall (moving lid)
        for (int i = 1; i < nx-1; ++i) {
            const int top_idx = i + (ny-1)*nx;
            vorticity[top_idx] = -2.0*streamFunction[top_idx-nx]/(dy*dy) - 2.0*U/dy;
        }
        // Bottom wall
        for (int i = 1; i < nx-1; ++i) {
            vorticity[i] = -2.0*streamFunction[i+nx]/(dy*dy);
        }
        // Vertical walls
        for (int j = 1; j < ny-1; ++j) {
            const int left_idx = j*nx;
            vorticity[left_idx] = -2.0*streamFunction[left_idx+1]/(dx*dx);
            const int right_idx = (j+1)*nx-1;
            vorticity[right_idx] = -2.0*streamFunction[right_idx-1]/(dx*dx);
        }

        // Corner averaging
        vorticity[0] = 0.5*(vorticity[1] + vorticity[nx]);
        vorticity[nx-1] = 0.5*(vorticity[nx-2] + vorticity[2*nx-1]);
        vorticity[(ny-1)*nx] = 0.5*(vorticity[(ny-2)*nx] + vorticity[(ny-1)*nx+1]);
        vorticity[ny*nx-1] = 0.5*(vorticity[ny*nx-2] + vorticity[(ny-1)*nx-1]);
    }

    VectorObj<TNum> computeVorticityDerivative(const VectorObj<TNum>& w) const {
        VectorObj<TNum> dwdt(w.size(), 0.0);
        const TNum re_inv = 1.0/Re;
        const TNum dx2 = dx*dx;
        const TNum dy2 = dy*dy;

        for (int j = 1; j < ny-1; ++j) {
            for (int i = 1; i < nx-1; ++i) {
                const int idx = i + j*nx;
                
                const TNum dwdx = (w[idx+1] - w[idx-1])/(2.0*dx);
                const TNum dwdy = (w[idx+nx] - w[idx-nx])/(2.0*dy);
                
                const TNum conv = u_velocity[idx]*dwdx + v_velocity[idx]*dwdy;
                const TNum diff = (w[idx+1] - 2*w[idx] + w[idx-1])/dx2
                                + (w[idx+nx] - 2*w[idx] + w[idx-nx])/dy2;
                
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
        for (size_t i = 0; i < vorticity.size(); ++i) {
            max_err = std::max(max_err, std::abs(vorticity[i]-prev[i]));
        }
        return max_err < tolerance;
    }

public:
    VorticityStreamSolver(int nx_, int ny_, TNum Re_, TNum U_ = 1.0,
                         TNum tol = 1e-6, TNum cfl = 0.5)
        : nx(nx_), ny(ny_), Re(Re_), U(U_), tolerance(tol), cfl_limit(cfl) {  // Initialize preconditioner with Laplacian

        if (nx < 3 || ny < 3) throw std::invalid_argument("Grid size must be at least 3x3");
        if (Re <= 0) throw std::invalid_argument("Reynolds number must be positive");

        dx = 1.0/(nx-1);
        dy = 1.0/(ny-1);

        vorticity = VectorObj<TNum>(nx*ny, 0.0);
        streamFunction = VectorObj<TNum>(nx*ny, 0.0);
        u_velocity = VectorObj<TNum>(nx*ny, 0.0);
        v_velocity = VectorObj<TNum>(nx*ny, 0.0);

        buildLaplacianMatrix();
        updateBoundaryConditions();
    }

    bool solve(TNum dt, int max_steps) {
        RungeKutta<TNum, VectorObj<TNum>> rk4;
        VectorObj<TNum> poisson_rhs(nx*ny);

        gmres.enablePreconditioner();
        for (int step = 0; step < max_steps; ++step) {
            // Solve Poisson equation: ∇²ψ = -ω
            poisson_rhs = vorticity * -1.0;
            gmres.solve(laplacian, poisson_rhs, streamFunction,
                            gmres_max_iter, gmres_krylov_dim, gmres_tol) ;

            computeVelocityField();
            updateBoundaryConditions();

            // Calculate adaptive time step
            const TNum cfl = computeCFL(dt);
            if (cfl > cfl_limit) {
                std::cout << "CFL violation: " << cfl << " > " << cfl_limit 
                          << " at step " << step << std::endl;
                dt *= 0.8 * cfl_limit / cfl;
                continue;
            }

            // Time integration
            const VectorObj<TNum> old_vorticity = vorticity;
            auto rhs_func = [this](const auto& w) { return computeVorticityDerivative(w); };
            rk4.solve(vorticity, rhs_func, dt, 20);

            // Convergence check
            if (step % 100 == 0) {
                std::cout << "Step " << step << ", CFL = " << cfl 
                          << ", dt = " << dt << std::endl;
                if (checkConvergence(old_vorticity)) {
                    std::cout << "Converged after " << step << " steps\n";
                    return true;
                }
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
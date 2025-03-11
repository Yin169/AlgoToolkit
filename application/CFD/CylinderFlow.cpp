#include "../../src/PDEs/FDM/NavierStoke.hpp"
#include "../../src/PDEs/Mesh/Adaptor.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <iomanip>

// Define the type we'll use for calculations
using TNum = double;

// Function to save velocity and pressure fields to VTK file for visualization
void saveToVTK(const std::string& filename, 
               const VectorObj<TNum>& u, const VectorObj<TNum>& v, const VectorObj<TNum>& w,
               const VectorObj<TNum>& p, int nx, int ny, int nz,
               const MeshAdapter<TNum>& mesh_adapter) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    // Write VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "Cylinder Flow Simulation\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_GRID\n";
    file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
    file << "POINTS " << nx * ny * nz << " float\n";
    
    // Write grid points (physical coordinates)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                auto coords = mesh_adapter.getPhysicalCoordinates(i, j, k);
                file << coords[0] << " " << coords[1] << " " << coords[2] << "\n";
            }
        }
    }
    
    // Write velocity field
    file << "POINT_DATA " << nx * ny * nz << "\n";
    
    // Write u component
    file << "SCALARS u float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                file << u[idx] << "\n";
            }
        }
    }
    
    // Write v component
    file << "SCALARS v float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                file << v[idx] << "\n";
            }
        }
    }
    
    // Write w component
    file << "SCALARS w float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                file << w[idx] << "\n";
            }
        }
    }
    
    // Write pressure
    file << "SCALARS pressure float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                file << p[idx] << "\n";
            }
        }
    }
    
    // Write velocity vector field
    file << "VECTORS velocity float\n";
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + j * nx + k * nx * ny;
                file << u[idx] << " " << v[idx] << " " << w[idx] << "\n";
            }
        }
    }
    
    // Calculate vorticity (only for 2D case, assuming k=0)
    if (nz == 1) {
        file << "SCALARS vorticity float 1\n";
        file << "LOOKUP_TABLE default\n";
        
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                TNum vort = 0.0;
                
                // Skip boundary points
                if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
                    int idx = i + j * nx;
                    int idx_ip = (i+1) + j * nx;
                    int idx_im = (i-1) + j * nx;
                    int idx_jp = i + (j+1) * nx;
                    int idx_jm = i + (j-1) * nx;
                    
                    // Get physical coordinates for grid spacing
                    auto coords = mesh_adapter.getPhysicalCoordinates(i, j, 0);
                    auto coords_ip = mesh_adapter.getPhysicalCoordinates(i+1, j, 0);
                    auto coords_im = mesh_adapter.getPhysicalCoordinates(i-1, j, 0);
                    auto coords_jp = mesh_adapter.getPhysicalCoordinates(i, j+1, 0);
                    auto coords_jm = mesh_adapter.getPhysicalCoordinates(i, j-1, 0);
                    
                    // Calculate physical grid spacing
                    TNum dx = (coords_ip[0] - coords_im[0]) / 2.0;
                    TNum dy = (coords_jp[1] - coords_jm[1]) / 2.0;
                    
                    // Calculate vorticity (dv/dx - du/dy)
                    vort = (v[idx_ip] - v[idx_im]) / (2.0 * dx) - 
                           (u[idx_jp] - u[idx_jm]) / (2.0 * dy);
                }
                
                file << vort << "\n";
            }
        }
    }
    
    file.close();
}

int main() {
    // Simulation parameters
    const int nx = 128;       // Grid points in radial direction
    const int ny = 128;       // Grid points in angular direction
    const int nz = 3;         // 2D simulation (single layer in z)
    
    const TNum Re = 200.0;    // Reynolds number
    const TNum U_inf = 1.0;   // Free stream velocity
    const TNum D = 1.0;       // Cylinder diameter
    
    // Physical domain dimensions
    const TNum r_min = 0.5;   // Cylinder radius (D/2)
    const TNum r_max = 10.0;  // Outer boundary radius
    const TNum theta_min = 0.0;
    const TNum theta_max = 2.0 * M_PI;
    const TNum z_min = 0.0;
    const TNum z_max = 0.1;   // Small z-extent for 2D simulation
    
    // Time stepping parameters
    const TNum dt_init = 0.0001;
    const TNum t_end = 100.0;
    const int output_freq = 100;
    
    std::cout << "Setting up cylinder flow simulation..." << std::endl;
    
    // Create mesh adapter with cylindrical coordinates
    MeshAdapter<TNum> mesh_adapter(nx, ny, nz, MeshType::CYLINDRICAL,
                                  r_min, r_max, theta_min, theta_max, z_min, z_max);
                                  
    // Create Navier-Stokes solver
    auto solver = mesh_adapter.createSolver(dt_init, Re);
    
    // Set time integration method and advection scheme
    solver->setTimeIntegrationMethod(TimeIntegration::RK4);
    solver->setAdvectionScheme(AdvectionScheme::CENTRAL);
    
    // Enable adaptive time stepping
    // solver->enableAdaptiveTimeStep(0.3);
    
    // Define boundary conditions in physical space
    // For cylinder surface (r = r_min)
    auto u_bc_phys = [r_min, r_max, U_inf](TNum x, TNum y, TNum z, TNum t) -> TNum {
        // Calculate r and theta from Cartesian coordinates
        TNum r = std::sqrt(x*x + y*y);
        TNum theta = std::atan2(y, x);
        
        if (std::abs(r - r_min) < 1e-6) {
            // No-slip condition on cylinder surface
            return 0.0;
        } else if (std::abs(r - r_max) < 1e-6) {
            // Free stream at outer boundary
            return U_inf;
        } else {
            // Interior point - will be computed
            return 0.0;
        }
    };
    
    auto v_bc_phys = [r_min, r_max, U_inf](TNum x, TNum y, TNum z, TNum t) -> TNum {
        // Calculate r and theta from Cartesian coordinates
        TNum r = std::sqrt(x*x + y*y);
        TNum theta = std::atan2(y, x);
        
        if (std::abs(r - r_min) < 1e-6) {
            // No-slip condition on cylinder surface
            return 0.0;
        } else if (std::abs(r - r_max) < 1e-6) {
            // Free stream at outer boundary
            return 0.0;
        } else {
            // Interior point - will be computed
            return 0.0;
        }
    };
    
    auto w_bc_phys = [](TNum x, TNum y, TNum z, TNum t) -> TNum {
        // Zero w-component for 2D simulation
        return 0.0;
    };
    
    // Adapt boundary conditions to computational space
    auto u_bc_comp = mesh_adapter.adaptBoundaryCondition(u_bc_phys, 0);
    auto v_bc_comp = mesh_adapter.adaptBoundaryCondition(v_bc_phys, 1);
    auto w_bc_comp = mesh_adapter.adaptBoundaryCondition(w_bc_phys, 2);
    
    // Set boundary conditions
    solver->setBoundaryConditions(u_bc_comp, v_bc_comp, w_bc_comp);
    
    // Initialize velocity fields
    int n = nx * ny * nz;
    VectorObj<TNum> u_phys(n, 0.0);
    VectorObj<TNum> v_phys(n, 0.0);
    VectorObj<TNum> w_phys(n, 0.0);
    
    // Initialize with free stream velocity in physical space
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                auto coords = mesh_adapter.getPhysicalCoordinates(i, j, k);
                TNum x = coords[0];
                TNum y = coords[1];
                TNum r = std::sqrt(x*x + y*y);
                TNum theta = std::atan2(y, x);
                
                int idx = i + j * nx + k * nx * ny;
                
                if (r > r_min) {
                    // Initialize with potential flow around cylinder
                    // u = U_inf * (1 - (r_min*r_min)/(r*r)) * cos(theta)
                    // v = -U_inf * (1 + (r_min*r_min)/(r*r)) * sin(theta)
                    u_phys[idx] = U_inf * (1.0 - (r_min*r_min)/(r*r)) * std::cos(theta);
                    v_phys[idx] = -U_inf * (1.0 + (r_min*r_min)/(r*r)) * std::sin(theta);
                } else {
                    // Inside cylinder (should not occur with proper mesh)
                    u_phys[idx] = 0.0;
                    v_phys[idx] = 0.0;
                }
                
                w_phys[idx] = 0.0;  // 2D simulation
            }
        }
    }
    
    // Adapt velocity field to computational space
    VectorObj<TNum> u_comp(n), v_comp(n), w_comp(n);
    mesh_adapter.adaptVelocityField(u_phys, v_phys, w_phys, u_comp, v_comp, w_comp);
    
    // Set initial conditions
    solver->setInitialConditions(u_comp, v_comp, w_comp);
    
    // Output callback function
    int step_count = 0;
    auto output_callback = [&](const VectorObj<TNum>& u, const VectorObj<TNum>& v, 
                              const VectorObj<TNum>& w, const VectorObj<TNum>& p, TNum time) {
        // Convert velocity back to physical space
        VectorObj<TNum> u_phys_out(n), v_phys_out(n), w_phys_out(n);
        mesh_adapter.convertVelocityToPhysical(u, v, w, u_phys_out, v_phys_out, w_phys_out);
        
        // Save to VTK file
        std::string filename = "cylinder_flow_" + std::to_string(step_count) + ".vtk";
        saveToVTK(filename, u_phys_out, v_phys_out, w_phys_out, p, nx, ny, nz, mesh_adapter);
        
        // Calculate and print some statistics
        TNum max_vel = 0.0;
        for (int i = 0; i < n; ++i) {
            TNum vel_mag = std::sqrt(u_phys_out[i]*u_phys_out[i] + v_phys_out[i]*v_phys_out[i]);
            max_vel = std::max(max_vel, vel_mag);
        }
        
        // Calculate drag and lift coefficients (simplified)
        TNum drag = 0.0, lift = 0.0;
        for (int j = 0; j < ny; ++j) {
            int i = 0;  // First radial point (cylinder surface)
            int idx = i + j * nx;
            
            auto coords = mesh_adapter.getPhysicalCoordinates(i, j, 0);
            TNum x = coords[0];
            TNum y = coords[1];
            TNum theta = std::atan2(y, x);
            
            // Calculate pressure force components
            TNum p_force = p[idx];  // Pressure at cylinder surface
            
            // Project pressure force to get drag and lift components
            drag += p_force * std::cos(theta) * r_min * (theta_max / ny);
            lift += p_force * std::sin(theta) * r_min * (theta_max / ny);
        }
        
        // Normalize by dynamic pressure and cylinder diameter
        TNum dynamic_pressure = 0.5 * U_inf * U_inf;
        TNum Cd = drag / (dynamic_pressure * D);
        TNum Cl = lift / (dynamic_pressure * D);
        
        std::cout << "Time: " << std::fixed << std::setprecision(3) << time
                  << ", Max velocity: " << std::setprecision(3) << max_vel
                  << ", Cd: " << std::setprecision(4) << Cd
                  << ", Cl: " << std::setprecision(4) << Cl << std::endl;
        
        step_count++;
    };
    
    // Run simulation
    std::cout << "Starting simulation..." << std::endl;
    solver->solve(0.0, t_end, output_freq, output_callback);
    
    std::cout << "Simulation completed." << std::endl;
    
    return 0;
}
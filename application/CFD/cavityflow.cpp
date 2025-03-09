#include "../../src/PDEs/SU2Mesh/readMesh.hpp"
#include "../../src/PDEs/FDM/NavierStoke.hpp"
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>

int main(int argc, char** argv) {
    std::string mesh_file = "/Users/yincheangng/worksapce/Github/MyAlgorithmicToolkit/meshfile/mesh_cavity_257x257.su2";
    
    if (argc > 1) {
        mesh_file = argv[1];
    }
    
    if (!std::filesystem::exists(mesh_file)) {
        std::cerr << "Error: Mesh file " << mesh_file << " does not exist." << std::endl;
        std::cerr << "Usage: " << argv[0] << " [mesh_file.su2]" << std::endl;
        return 1;
    }
    
    std::cout << "Starting lid-driven cavity flow simulation..." << std::endl;
    std::cout << "Mesh file: " << mesh_file << std::endl;
    
    try {
        // Simulation parameters
        double dt = 0.001;       // Time step
        double Re = 1000.0;      // Reynolds number
        double duration = 20.0;  // Total simulation time
        int output_freq = 100;   // Output frequency (in time steps)
        int vtk_freq = 1000;     // VTK output frequency (in time steps)
        
        std::cout << "Parameters:" << std::endl;
        std::cout << "  Reynolds number: " << Re << std::endl;
        std::cout << "  Time step: " << dt << std::endl;
        std::cout << "  Duration: " << duration << std::endl;
        
        // Create solver with SU2 mesh
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "Reading mesh..." << std::endl;
        SU2Mesh mesh;
        if (!mesh.readMeshFile(mesh_file)) {
            throw std::runtime_error("Failed to read SU2 mesh file");
        }
        
        if (!mesh.isStructured()) {
            throw std::runtime_error("NavierStokes solver requires structured grid");
        }
        
        // Create solver with mesh dimensions
        int nx = mesh.getNx();
        int ny = mesh.getNy();
        int nz = mesh.getDimension() == 3 ? mesh.getNz() : 1;
        double dx = mesh.getDx();
        double dy = mesh.getDy();
        double dz = mesh.getDimension() == 3 ? mesh.getDz() : 1.0;
        
        std::cout << "Initializing solver..." << std::endl;
        NavierStokesSolver3D<double> ns_solver(nx, ny, nz, dx, dy, dz, dt, Re);
        
        // Set boundary conditions for lid-driven cavity
        std::cout << "Setting boundary conditions..." << std::endl;
        
        // Top wall moving with u=1.0 (lid-driven cavity)
        ns_solver.setBoundaryConditions(
            // u velocity boundary conditions
            [](double x, double y, double z, double t) {
                // Top wall (y = max)
                if (std::abs(y - 1.0) < 1e-6) return 1.0;
                // All other walls
                return 0.0;
            },
            // v velocity boundary conditions
            [](double x, double y, double z, double t) {
                return 0.0; // No flow in y direction at boundaries
            },
            // w velocity boundary conditions
            [](double x, double y, double z, double t) {
                return 0.0; // No flow in z direction at boundaries
            }
        );
        
        // Set initial conditions (zero velocity everywhere)
        std::cout << "Setting initial conditions..." << std::endl;
        ns_solver.setInitialConditions(
            [](double x, double y, double z) { return 0.0; },  // u
            [](double x, double y, double z) { return 0.0; },  // v
            [](double x, double y, double z) { return 0.0; }   // w
        );
        
        // Create output directory
        std::string output_dir = "/Users/yincheangng/worksapce/Github/MyAlgorithmicToolkit/output/cavity_flow";
        std::filesystem::create_directories(output_dir);
        
        // Run simulation with manual time stepping to allow for monitoring and VTK output
        int num_steps = static_cast<int>(duration / dt);
        
        std::cout << "Starting simulation for " << num_steps << " time steps..." << std::endl;
        
        for (int step = 0; step < num_steps; step++) {
            double current_time = step * dt;
            
            // Advance one time step
            ns_solver.step(current_time);
            
            // Output progress
            if (step % output_freq == 0) {
                double kinetic_energy = ns_solver.calculateKineticEnergy();
                auto current_time_point = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    current_time_point - start_time).count();
                
                std::cout << "Step: " << std::setw(8) << step 
                          << " | Time: " << std::fixed << std::setprecision(3) << current_time 
                          << " | Kinetic Energy: " << std::scientific << std::setprecision(6) << kinetic_energy
                          << " | Elapsed: " << elapsed << "s" << std::endl;
                
                // Calculate max divergence (should be close to zero for incompressible flow)
                VectorObj<double> div = ns_solver.calculateDivergence();
                double max_div = 0.0;
                for (int i = 0; i < div.size(); i++) {
                    max_div = std::max(max_div, std::abs(div[i]));
                }
                
                std::cout << "  Max Divergence: " << std::scientific << max_div << std::endl;
            }
            
            // Output VTK file for visualization
            if (step % vtk_freq == 0 || step == num_steps - 1) {
                std::string vtk_file = output_dir + "/cavity_flow_" + std::to_string(step) + ".vtk";
                std::cout << "Writing VTK file: " << vtk_file << std::endl;
                
                // Export solution to VTK format
                std::ofstream file(vtk_file);
                if (file.is_open()) {
                    // Write VTK header
                    file << "# vtk DataFile Version 3.0\n";
                    file << "Cavity Flow Solution\n";
                    file << "ASCII\n";
                    file << "DATASET STRUCTURED_GRID\n";
                    file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
                    file << "POINTS " << nx * ny * nz << " float\n";
                    
                    // Write point coordinates
                    for (int k = 0; k < nz; k++) {
                        for (int j = 0; j < ny; j++) {
                            for (int i = 0; i < nx; i++) {
                                double x = i * dx;
                                double y = j * dy;
                                double z = k * dz;
                                file << x << " " << y << " " << z << "\n";
                            }
                        }
                    }
                    
                    // Write point data
                    file << "POINT_DATA " << nx * ny * nz << "\n";
                    
                    // Write velocity vector field
                    file << "VECTORS velocity float\n";
                    for (int idx = 0; idx < nx * ny * nz; idx++) {
                        file << ns_solver.getU()[idx] << " " 
                             << ns_solver.getV()[idx] << " " 
                             << ns_solver.getW()[idx] << "\n";
                    }
                    
                    // Write pressure scalar field
                    file << "SCALARS pressure float 1\n";
                    file << "LOOKUP_TABLE default\n";
                    for (int idx = 0; idx < nx * ny * nz; idx++) {
                        file << ns_solver.getP()[idx] << "\n";
                    }
                    
                    file.close();
                }
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        
        std::cout << "Simulation completed in " << elapsed << " seconds." << std::endl;
        std::cout << "Output files written to " << output_dir << std::endl;
        
        // Calculate vorticity for final state
        VectorObj<double> vort_x, vort_y, vort_z;
        ns_solver.calculateVorticity(vort_x, vort_y, vort_z);
        
        // Export final solution with vorticity
        std::string final_vtk = output_dir + "/cavity_flow_final.vtk";
        std::ofstream final_file(final_vtk);
        if (final_file.is_open()) {
            // Write VTK header
            final_file << "# vtk DataFile Version 3.0\n";
            final_file << "Cavity Flow Final Solution with Vorticity\n";
            final_file << "ASCII\n";
            final_file << "DATASET STRUCTURED_GRID\n";
            final_file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
            final_file << "POINTS " << nx * ny * nz << " float\n";
            
            // Write point coordinates
            for (int k = 0; k < nz; k++) {
                for (int j = 0; j < ny; j++) {
                    for (int i = 0; i < nx; i++) {
                        double x = i * dx;
                        double y = j * dy;
                        double z = k * dz;
                        final_file << x << " " << y << " " << z << "\n";
                    }
                }
            }
            
            // Write point data
            final_file << "POINT_DATA " << nx * ny * nz << "\n";
            
            // Write velocity vector field
            final_file << "VECTORS velocity float\n";
            for (int idx = 0; idx < nx * ny * nz; idx++) {
                final_file << ns_solver.getU()[idx] << " " 
                         << ns_solver.getV()[idx] << " " 
                         << ns_solver.getW()[idx] << "\n";
            }
            
            // Write pressure scalar field
            final_file << "SCALARS pressure float 1\n";
            final_file << "LOOKUP_TABLE default\n";
            for (int idx = 0; idx < nx * ny * nz; idx++) {
                final_file << ns_solver.getP()[idx] << "\n";
            }
            
            // Write vorticity vector field
            final_file << "VECTORS vorticity float\n";
            for (int idx = 0; idx < nx * ny * nz; idx++) {
                final_file << vort_x[idx] << " " 
                         << vort_y[idx] << " " 
                         << vort_z[idx] << "\n";
            }
            
            final_file.close();
        }
        
        std::cout << "Final solution exported to " << final_vtk << std::endl;
        std::cout << "You can visualize the results using ParaView or other VTK viewers." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
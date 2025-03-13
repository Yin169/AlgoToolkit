#include "../../src/PDEs/FFT/FastNavierStoke.hpp"
#include "../../src/Obj/VectorObj.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <memory>

// Taylor-Green vortex test case
template <typename TNum>
class TaylorGreenVortexTest {
private:
    SpectralNavierStokesSolver<TNum> solver;
    int nx, ny, nz;
    TNum Lx, Ly, Lz;
    TNum Re;
    TNum dt;
    std::string output_dir;
    
    // For tracking simulation statistics
    std::vector<TNum> time_history;
    std::vector<TNum> energy_history;
    std::vector<TNum> enstrophy_history;
    std::vector<TNum> max_divergence_history;
    
public:
    TaylorGreenVortexTest(int nx_, int ny_, int nz_,
                         TNum Lx_ = 2.0 * M_PI, TNum Ly_ = 2.0 * M_PI, TNum Lz_ = 2.0 * M_PI,
                         TNum Re_ = 1600.0, TNum dt_ = 0.001,
                         const std::string& output_dir_ = "./output_tgv")
        : solver(nx_, ny_, nz_, Lx_, Ly_, Lz_, dt_, Re_),
          nx(nx_), ny(ny_), nz(nz_),
          Lx(Lx_), Ly(Ly_), Lz(Lz_),
          Re(Re_), dt(dt_),
          output_dir(output_dir_) {
        
        // Initialize with Taylor-Green vortex
        initializeTaylorGreenVortex();
        
        // Set time integration method (RK4 is more accurate but slower)
        solver.setTimeIntegrationMethod(SpectralTimeIntegration::RK4);
        
        // Enable adaptive time stepping for stability
        solver.enableAdaptiveTimeStep(0.5);
    }
    
    // Initialize Taylor-Green vortex flow
    void initializeTaylorGreenVortex() {
        VectorObj<TNum> u0(nx * ny * nz);
        VectorObj<TNum> v0(nx * ny * nz);
        VectorObj<TNum> w0(nx * ny * nz);
        
        TNum dx = Lx / nx;
        TNum dy = Ly / ny;
        TNum dz = Lz / nz;
        
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    TNum x = i * dx;
                    TNum y = j * dy;
                    TNum z = k * dz;
                    
                    int index = i + j * nx + k * nx * ny;
                    
                    // Taylor-Green vortex initial condition
                    u0[index] = sin(x) * cos(y) * cos(z);
                    v0[index] = -cos(x) * sin(y) * cos(z);
                    w0[index] = 0.0;
                }
            }
        }
        
        solver.setInitialConditions(u0, v0, w0);
    }
    
    // Run the simulation
    void run(int num_steps, int output_frequency = 100) {
        std::cout << "Starting Taylor-Green vortex simulation at Re = " << Re << std::endl;
        std::cout << "Grid: " << nx << " x " << ny << " x " << nz << std::endl;
        std::cout << "Initial dt: " << dt << std::endl;
        
        // Create output directory if it doesn't exist
        std::string cmd = "mkdir -p " + output_dir;
        system(cmd.c_str());
        
        // Open statistics file
        std::ofstream stats_file(output_dir + "/statistics.csv");
        stats_file << "time,energy,enstrophy,max_divergence,dt" << std::endl;
        
        // Setup output callback
        auto output_callback = [this, &stats_file](const VectorObj<TNum>& u, const VectorObj<TNum>& v, const VectorObj<TNum>& w, TNum time, TNum tstep) {
            // Calculate statistics
            TNum energy = solver.calculateKineticEnergy();
            TNum enstrophy = solver.calculateEnstrophy();
            
            // Calculate maximum divergence (should be close to zero)
            VectorObj<TNum> div = solver.calculateDivergence();
            TNum max_div = 0.0;
            for (size_t i = 0; i < div.size(); i++) {
                max_div = std::max(max_div, std::abs(div[i]));
            }
            
            // Store statistics
            time_history.push_back(time);
            energy_history.push_back(energy);
            enstrophy_history.push_back(enstrophy);
            max_divergence_history.push_back(max_div);
            
            // Write to statistics file
            stats_file << time << "," << energy << "," << enstrophy << "," 
                      << max_div << "," << solver.getDt() << std::endl;
            
            // Output progress
            std::cout << "Time: " << std::fixed << std::setprecision(5) << time
                      << ", Energy: " << std::setprecision(10) << energy
                      << ", Enstrophy: " << enstrophy
                      << ", Max Div: " << std::scientific << max_div
                      << ", dt: " << solver.getDt() << std::endl;
            
            // Save velocity field
            saveVelocityField(tstep);
            
            // Save vorticity field
            saveVorticityField(tstep);
        };
        
        // Start timer
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Run simulation
        solver.solve(num_steps, output_frequency, output_callback);
        
        // End timer
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        
        std::cout << "Simulation completed in " << duration << " seconds" << std::endl;
        stats_file.close();
        
        // Save final state
        saveVelocityField(time_history.back(), true);
        saveVorticityField(time_history.back(), true);

    }
    
private:
    // Save velocity field to VTK file for visualization
    void saveVelocityField(TNum time, bool is_final = false) {
        std::string filename;
        if (is_final) {
            filename = output_dir + "/velocity_final.vtk";
        } else {
            std::stringstream ss;
            ss << output_dir << "/velocity_" << time << ".vtk";
            filename = ss.str();
        }
        
        std::ofstream file(filename);
        
        // VTK file header
        file << "# vtk DataFile Version 3.0" << std::endl;
        file << "Taylor-Green Vortex Velocity Field at time " << time << std::endl;
        file << "ASCII" << std::endl;
        file << "DATASET STRUCTURED_POINTS" << std::endl;
        file << "DIMENSIONS " << nx << " " << ny << " " << nz << std::endl;
        file << "ORIGIN 0 0 0" << std::endl;
        file << "SPACING " << Lx/nx << " " << Ly/ny << " " << Lz/nz << std::endl;
        file << "POINT_DATA " << nx * ny * nz << std::endl;
        
        // Write velocity vector field
        file << "VECTORS velocity float" << std::endl;
        const VectorObj<TNum>& u = solver.getU();
        const VectorObj<TNum>& v = solver.getV();
        const VectorObj<TNum>& w = solver.getW();
        
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    int index = i + j * nx + k * nx * ny;
                    file << u[index] << " " << v[index] << " " << w[index] << std::endl;
                }
            }
        }
        
        file.close();
    }
    
    // Save vorticity field to VTK file for visualization
    void saveVorticityField(TNum time, bool is_final = false) {
        std::string filename;
        if (is_final) {
            filename = output_dir + "/vorticity_final.vtk";
        } else {
            std::stringstream ss;
            ss << output_dir << "/vorticity_" << time << ".vtk";
            filename = ss.str();
        }
        
        std::ofstream file(filename);
        
        // Get vorticity field
        VectorObj<TNum> omega_x(nx * ny * nz);
        VectorObj<TNum> omega_y(nx * ny * nz);
        VectorObj<TNum> omega_z(nx * ny * nz);
        solver.getVorticityField(omega_x, omega_y, omega_z);
        
        // VTK file header
        file << "# vtk DataFile Version 3.0" << std::endl;
        file << "Taylor-Green Vortex Vorticity Field at time " << time << std::endl;
        file << "ASCII" << std::endl;
        file << "DATASET STRUCTURED_POINTS" << std::endl;
        file << "DIMENSIONS " << nx << " " << ny << " " << nz << std::endl;
        file << "ORIGIN 0 0 0" << std::endl;
        file << "SPACING " << Lx/nx << " " << Ly/ny << " " << Lz/nz << std::endl;
        file << "POINT_DATA " << nx * ny * nz << std::endl;
        
        // Write vorticity vector field
        file << "VECTORS vorticity float" << std::endl;
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    int index = i + j * nx + k * nx * ny;
                    file << omega_x[index] << " " << omega_y[index] << " " << omega_z[index] << std::endl;
                }
            }
        }
        
        // Also write vorticity magnitude
        file << "SCALARS vorticity_magnitude float" << std::endl;
        file << "LOOKUP_TABLE default" << std::endl;
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    int index = i + j * nx + k * nx * ny;
                    TNum magnitude = std::sqrt(
                        omega_x[index] * omega_x[index] +
                        omega_y[index] * omega_y[index] +
                        omega_z[index] * omega_z[index]
                    );
                    file << magnitude << std::endl;
                }
            }
        }
        
        file.close();
    }
};

// Main function to run the test case
int main(int argc, char** argv) {
    // Default parameters
    int nx = 64;  // Grid resolution in x
    int ny = 64;  // Grid resolution in y
    int nz = 64;  // Grid resolution in z
    double Re = 1600.0;  // Reynolds number
    double dt = 0.001;   // Initial time step
    int num_steps = 5000;  // Number of time steps
    int output_frequency = 100;  // Output every 100 steps
    
    // Create and run the Taylor-Green vortex test
    TaylorGreenVortexTest<double> test(nx, ny, nz, 2.0 * M_PI, 2.0 * M_PI, 2.0 * M_PI, Re, dt);
    test.run(num_steps, output_frequency);
    
    std::cout << "Taylor-Green vortex simulation completed successfully!" << std::endl;
    std::cout << "Results saved to ./output_tgv directory" << std::endl;
    std::cout << "To visualize the results, run the generated Python script or use ParaView to open the VTK files" << std::endl;
    
    return 0;
}
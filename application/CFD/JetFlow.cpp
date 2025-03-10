#include "../../src/PDEs/FDM/NavierStoke.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <vector>

// Function to save velocity field to a file in VTK format
template <typename TNum>
void saveVTKFile(const NavierStokesSolver3D<TNum>& solver, const std::string& filename) {
    // Existing code...
    int nx = solver.getNx();
    int ny = solver.getNy();
    int nz = solver.getNz();
    TNum dx = solver.getDx();
    TNum dy = solver.getDy();
    TNum dz = solver.getDz();
    
    std::ofstream outFile(filename);
    outFile << "# vtk DataFile Version 3.0\n";
    outFile << "Jet Flow Simulation\n";
    outFile << "ASCII\n";
    outFile << "DATASET STRUCTURED_GRID\n";
    outFile << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
    outFile << "POINTS " << nx*ny*nz << " float\n";
    
    // Write grid points
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                outFile << i*dx << " " << j*dy << " " << k*dz << "\n";
            }
        }
    }
    
    // Write velocity vectors
    outFile << "POINT_DATA " << nx*ny*nz << "\n";
    outFile << "VECTORS velocity float\n";
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                outFile << solver.getU(i, j, k) << " " 
                        << solver.getV(i, j, k) << " " 
                        << solver.getW(i, j, k) << "\n";
            }
        }
    }
    
    // Write velocity magnitude
    outFile << "SCALARS velocity_magnitude float 1\n";
    outFile << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                TNum u = solver.getU(i, j, k);
                TNum v = solver.getV(i, j, k);
                TNum w = solver.getW(i, j, k);
                TNum magnitude = std::sqrt(u*u + v*v + w*w);
                outFile << magnitude << "\n";
            }
        }
    }
    
    // Write pressure field
    outFile << "SCALARS pressure float 1\n";
    outFile << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                outFile << solver.getP(i, j, k) << "\n";
            }
        }
    }
    
    // Calculate and write vorticity
    VectorObj<TNum> vort_x, vort_y, vort_z;
    solver.calculateVorticity(vort_x, vort_y, vort_z);
    
    // Vorticity vector field
    outFile << "VECTORS vorticity float\n";
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = i + j*nx + k*nx*ny;
                outFile << vort_x[idx] << " " << vort_y[idx] << " " << vort_z[idx] << "\n";
            }
        }
    }
    
    // Vorticity magnitude (Q-criterion for vortex identification)
    outFile << "SCALARS vorticity_magnitude float 1\n";
    outFile << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = i + j*nx + k*nx*ny;
                TNum magnitude = std::sqrt(vort_x[idx]*vort_x[idx] + 
                                          vort_y[idx]*vort_y[idx] + 
                                          vort_z[idx]*vort_z[idx]);
                outFile << magnitude << "\n";
            }
        }
    }
    
    // Calculate and write divergence (should be close to zero)
    VectorObj<TNum> div = solver.calculateDivergence();
    
    outFile << "SCALARS divergence float 1\n";
    outFile << "LOOKUP_TABLE default\n";
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = i + j*nx + k*nx*ny;
                outFile << div[idx] << "\n";
            }
        }
    }
    
    outFile.close();
}

// Function to parse command line arguments
template <typename T>
T getOption(char** begin, char** end, const std::string& option, T default_value) {
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        std::istringstream iss(*itr);
        T value;
        if (iss >> value) {
            return value;
        }
    }
    return default_value;
}

bool hasOption(char** begin, char** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    const int nx = getOption(argv, argv + argc, "--nx", 100);
    const int ny = getOption(argv, argv + argc, "--ny", 50);
    const int nz = getOption(argv, argv + argc, "--nz", 50);
    const double dx = getOption(argv, argv + argc, "--dx", 0.1);
    const double dy = getOption(argv, argv + argc, "--dy", 0.1);
    const double dz = getOption(argv, argv + argc, "--dz", 0.1);
    const double dt = getOption(argv, argv + argc, "--dt", 0.005);
    const double Re = getOption(argv, argv + argc, "--re", 1000.0);  // Reynolds number
    const double totalTime = getOption(argv, argv + argc, "--time", 20.0);
    const int saveInterval = getOption(argv, argv + argc, "--save", 50);  // Save every 50 steps
    
    // Jet parameters
    const double jetRadius = getOption(argv, argv + argc, "--radius", 1.0);
    const double jetVelocity = getOption(argv, argv + argc, "--velocity", 2.0);
    const double jetCenterY = getOption(argv, argv + argc, "--center-y", ny * dy / 2.0);
    const double jetCenterZ = getOption(argv, argv + argc, "--center-z", nz * dz / 2.0);
    
    // Advanced options
    const bool useTurbulence = hasOption(argv, argv + argc, "--turbulence");
    const bool useAdaptiveTimeStep = hasOption(argv, argv + argc, "--adaptive-dt");
    const bool usePulsatingJet = hasOption(argv, argv + argc, "--pulsating");
    const double pulsePeriod = getOption(argv, argv + argc, "--pulse-period", 5.0);
    const double pulseAmplitude = getOption(argv, argv + argc, "--pulse-amplitude", 0.5);
    
    // Print simulation parameters
    std::cout << "=== Jet Flow Simulation Parameters ===" << std::endl;
    std::cout << "Grid: " << nx << " x " << ny << " x " << nz << std::endl;
    std::cout << "Cell size: " << dx << " x " << dy << " x " << dz << std::endl;
    std::cout << "Reynolds number: " << Re << std::endl;
    std::cout << "Jet velocity: " << jetVelocity << std::endl;
    std::cout << "Jet radius: " << jetRadius << std::endl;
    std::cout << "Turbulence model: " << (useTurbulence ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Adaptive time stepping: " << (useAdaptiveTimeStep ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Pulsating jet: " << (usePulsatingJet ? "Enabled" : "Disabled") << std::endl;
    if (usePulsatingJet) {
        std::cout << "  Pulse period: " << pulsePeriod << std::endl;
        std::cout << "  Pulse amplitude: " << pulseAmplitude << std::endl;
    }
    std::cout << "======================================" << std::endl;
    
    // Create solver
    NavierStokesSolver3D<double> solver(nx, ny, nz, dx, dy, dz, dt, Re);
    
    // Enable advanced features
    if (useTurbulence) {
        solver.enableTurbulenceModel(true);
    }
    
    if (useAdaptiveTimeStep) {
        solver.setAdaptiveTimeStep(true, 0.5); // CFL number of 0.5
    }
    
    // Set pressure solver parameters for better convergence
    solver.setPressureSolverParams(5, 1e-6);
    
    // Define boundary conditions with convective outflow
    auto u_bc = [&solver, nx, ny, nz, dx, dy, dz, jetRadius, jetVelocity, jetCenterY, jetCenterZ, 
                 usePulsatingJet, pulsePeriod, pulseAmplitude]
               (double x, double y, double z, double t) -> double {
        // Jet inlet at x=0 (left boundary)
        if (std::abs(x) < 1e-10) {
            // Calculate distance from jet center
            double r = std::sqrt(std::pow(y - jetCenterY, 2) + std::pow(z - jetCenterZ, 2));
            
            if (r <= jetRadius) {
                // Smooth jet profile with cosine function for better stability
                double profile = 0.5 * (1.0 + std::cos(M_PI * r / jetRadius));
                
                // Apply pulsating effect if enabled
                double velocity = jetVelocity;
                if (usePulsatingJet) {
                    velocity *= (1.0 + pulseAmplitude * std::sin(2.0 * M_PI * t / pulsePeriod));
                }
                
                return velocity * profile;
            }
            return 0.0;
        }
        
        // Convective outflow at x=L (right boundary)
        if (std::abs(x - (nx-1)*dx) < 1e-10) {
            int i = nx-2;  // One cell before boundary
            int j = static_cast<int>(y/dy);
            int k = static_cast<int>(z/dz);
            
            // Ensure indices are within bounds
            j = std::max(0, std::min(j, ny-1));
            k = std::max(0, std::min(k, nz-1));
            
            return solver.getU(i, j, k);  // Zero gradient
        }
        
        // Free-slip at other boundaries
        return 0.0;
    };
    
    // v and w boundary conditions remain mostly the same
    auto v_bc = [&solver, nx, ny, nz, dx, dy, dz]
               (double x, double y, double z, double t) -> double {
        // No flow through inlet
        if (std::abs(x) < 1e-10) {
            return 0.0;
        }
        
        // Convective outflow at x=L (right boundary)
        if (std::abs(x - (nx-1)*dx) < 1e-10) {
            int i = nx-2;  // One cell before boundary
            int j = static_cast<int>(y/dy);
            int k = static_cast<int>(z/dz);
            
            // Ensure indices are within bounds
            j = std::max(0, std::min(j, ny-1));
            k = std::max(0, std::min(k, nz-1));
            
            return solver.getV(i, j, k);  // Zero gradient
        }
        
        // Free-slip at y boundaries
        if (std::abs(y) < 1e-10 || std::abs(y - (ny-1)*dy) < 1e-10) {
            return 0.0;  // No flow through boundary
        }
        
        // Free-slip at z boundaries
        if (std::abs(z) < 1e-10 || std::abs(z - (nz-1)*dz) < 1e-10) {
            return 0.0;  // No flow through boundary
        }
        
        return 0.0;
    };
    
    auto w_bc = [&solver, nx, ny, nz, dx, dy, dz]
               (double x, double y, double z, double t) -> double {
        // No flow through inlet
        if (std::abs(x) < 1e-10) {
            return 0.0;
        }
        
        // Convective outflow at x=L (right boundary)
        if (std::abs(x - (nx-1)*dx) < 1e-10) {
            int i = nx-2;  // One cell before boundary
            int j = static_cast<int>(y/dy);
            int k = static_cast<int>(z/dz);
            
            // Ensure indices are within bounds
            j = std::max(0, std::min(j, ny-1));
            k = std::max(0, std::min(k, nz-1));
            
            return solver.getW(i, j, k);  // Zero gradient
        }
        
        // Free-slip at y boundaries
        if (std::abs(y) < 1e-10 || std::abs(y - (ny-1)*dy) < 1e-10) {
            return 0.0;  // No flow through boundary
        }
        
        // Free-slip at z boundaries
        if (std::abs(z) < 1e-10 || std::abs(z - (nz-1)*dz) < 1e-10) {
            return 0.0;  // No flow through boundary
        }
        
        return 0.0;
    };
    
    // Set boundary conditions
    solver.setBoundaryConditions(u_bc, v_bc, w_bc);
    
    // Set initial conditions (small random perturbation for better vortex development)
    auto u_init = [jetVelocity, jetRadius, jetCenterY, jetCenterZ, dx, dy, dz]
                 (double x, double y, double z) -> double {
        // Small initial velocity in the domain
        double r = std::sqrt(std::pow(y - jetCenterY, 2) + std::pow(z - jetCenterZ, 2));
        
        if (x < 2.0*dx && r <= jetRadius) {
            // Smooth jet profile with cosine function
            double profile = 0.5 * (1.0 + std::cos(M_PI * r / jetRadius));
            return jetVelocity * profile;
        }
        
        // Small random perturbation elsewhere
        return 0.01 * jetVelocity * (2.0 * rand() / RAND_MAX - 1.0);
    };
    
    auto v_init = [jetVelocity](double x, double y, double z) -> double {
        // Small random perturbation
        return 0.01 * jetVelocity * (2.0 * rand() / RAND_MAX - 1.0);
    };
    
    auto w_init = [jetVelocity](double x, double y, double z) -> double {
        // Small random perturbation
        return 0.01 * jetVelocity * (2.0 * rand() / RAND_MAX - 1.0);
    };
    
    // Set initial conditions
    solver.setInitialConditions(u_init, v_init, w_init);
    
    // Add gravity or other external forces if needed
    // Example: Add a small buoyancy force in the y-direction
    solver.addForceField(
        [](double x, double y, double z, double t) -> double { return 0.0; },  // x-component
        [](double x, double y, double z, double t) -> double { return 0.05; }, // y-component
        [](double x, double y, double z, double t) -> double { return 0.0; },  // z-component
        0.0  // initial time
    );
    
    // Run simulation
    int numSteps = static_cast<int>(totalTime / dt);
    std::cout << "Starting jet flow simulation for " << numSteps << " steps..." << std::endl;
    
    // Create output directory
    system("mkdir -p jet_flow_results");
    
    // Track simulation time
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Variables to track simulation statistics
    double maxVelocity = 0.0;
    double minDt = dt;
    double maxDt = dt;
    
    // Main simulation loop
    for (int step = 0; step < numSteps; step++) {
        double currentTime = step * dt;
        
        // Update external forces if they're time-dependent
        if (usePulsatingJet) {
            solver.addForceField(
                [](double x, double y, double z, double t) -> double { return 0.0; },
                [](double x, double y, double z, double t) -> double { return 0.05; },
                [](double x, double y, double z, double t) -> double { return 0.0; },
                currentTime
            );
        }
        
        // Advance simulation by one time step
        solver.step(currentTime);
        
        // Track adaptive time step if enabled
        if (useAdaptiveTimeStep) {
            double currentDt = solver.getDt();
            minDt = std::min(minDt, currentDt);
            maxDt = std::max(maxDt, currentDt);
        }
        
        // Print progress and energy
        if (step % 100 == 0 || step == numSteps - 1) {
            double ke = solver.calculateKineticEnergy();
            
            // Calculate elapsed time and estimated time remaining
            auto currentClock = std::chrono::high_resolution_clock::now();
            auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(
                currentClock - startTime).count();
            double estimatedTotal = (elapsedTime * numSteps) / (step + 1.0);
            double remaining = estimatedTotal - elapsedTime;
            
            std::cout << "Step " << step << "/" << numSteps 
                      << ", Time: " << std::fixed << std::setprecision(3) << currentTime 
                      << ", Kinetic Energy: " << std::scientific << std::setprecision(6) << ke
                      << ", Elapsed: " << elapsedTime << "s"
                      << ", Remaining: " << remaining << "s";
            
            if (useAdaptiveTimeStep) {
                std::cout << ", dt range: [" << minDt << ", " << maxDt << "]";
            }
            
            std::cout << std::endl;
            
            // Check for divergence (optional stability check)
            VectorObj<double> div = solver.calculateDivergence();
            double max_div = 0.0;
            for (int i = 0; i < div.size(); i++) {
                max_div = std::max(max_div, std::abs(div[i]));
            }
            std::cout << "  Max Divergence: " << std::scientific << max_div << std::endl;
            
            // Check if simulation is stable
            if (std::isnan(ke) || ke > 1e10) {
                std::cerr << "ERROR: Simulation diverged at step " << step << std::endl;
                break;
            }
        }
        
        // Save results periodically
        if (step % saveInterval == 0 || step == numSteps - 1) {
            std::string filename = "jet_flow_results/jetflow_" + 
                                  std::to_string(step/saveInterval) + ".vtk";
            saveVTKFile(solver, filename);
        }
    }
    
    // Calculate total simulation time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalElapsed = std::chrono::duration_cast<std::chrono::seconds>(
        endTime - startTime).count();
    
    std::cout << "Simulation completed in " << totalElapsed << " seconds." << std::endl;
    std::cout << "Results saved to jet_flow_results directory." << std::endl;
    
    // Generate a simple Python script to visualize results with ParaView
    std::ofstream scriptFile("jet_flow_results/visualize.py");
    scriptFile << "# Run this script with: pvpython visualize.py\n";
    scriptFile << "from paraview.simple import *\n";
    scriptFile << "import os\n\n";
    scriptFile << "# Get all VTK files\n";
    scriptFile << "files = sorted([f for f in os.listdir('.') if f.endswith('.vtk')])\n\n";
    scriptFile << "# Create a reader for the VTK files\n";
    scriptFile << "reader = LegacyVTKReader(FileNames=files)\n";
    scriptFile << "Show(reader)\n";
    scriptFile << "# Create a better visualization\n";
    scriptFile << "renderView1 = GetActiveViewOrCreate('RenderView')\n";
    scriptFile << "# Display velocity vectors\n";
    scriptFile << "vectorDisplay = Show(reader, renderView1, 'GeometryRepresentation')\n";
    scriptFile << "vectorDisplay.SetRepresentationType('Volume')\n";
    scriptFile << "ColorBy(vectorDisplay, ('POINTS', 'velocity_magnitude'))\n";
    scriptFile << "# Add a color bar\n";
    scriptFile << "vectorDisplay.SetScalarBarVisibility(renderView1, True)\n";
    scriptFile << "# Set up a good camera angle\n";
    scriptFile << "renderView1.ResetCamera()\n";
    scriptFile << "renderView1.Update()\n";
    scriptFile << "# Save an animation\n";
    scriptFile << "SaveAnimation('jet_flow_animation.png', renderView1, ImageResolution=[1920, 1080],\n";
    scriptFile << "              FrameWindow=[0, len(files)-1])\n";
    scriptFile.close();
    
    std::cout << "Created visualization script: jet_flow_results/visualize.py" << std::endl;
    std::cout << "To visualize results, you can use ParaView or run: pvpython jet_flow_results/visualize.py" << std::endl;
    
    return 0;
}
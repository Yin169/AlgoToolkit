
#include "NavierStoke.hpp"
#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>

template <typename TNum>
class JetFlowSimulation {
private:
    NavierStokesSolver3D<TNum> solver;
    int nx, ny, nz;
    TNum dx, dy, dz;
    TNum jetVelocity;
    TNum jetRadius;
    TNum domainLength;
    TNum domainWidth;
    TNum domainHeight;
    
    // Output settings
    std::string outputDir;
    int outputFrequency;
    
    // Helper function to create output directory
    void createOutputDirectory() {
        std::string command = "mkdir -p " + outputDir;
        int result = system(command.c_str());
        if (result != 0) {
            std::cerr << "Failed to create output directory: " << outputDir << std::endl;
        }
    }
    
    // Write VTK file for visualization
    void writeVTKFile(int step, TNum time) {
        std::string filename = outputDir + "/jet_flow_" + std::to_string(step) + ".vtk";
        std::ofstream outFile(filename);
        
        if (!outFile.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }
        
        // VTK header
        outFile << "# vtk DataFile Version 3.0\n";
        outFile << "Jet Flow Simulation at time " << time << "\n";
        outFile << "ASCII\n";
        outFile << "DATASET STRUCTURED_GRID\n";
        outFile << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
        outFile << "POINTS " << nx * ny * nz << " float\n";
        
        // Grid points
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    outFile << i * dx << " " << j * dy << " " << k * dz << "\n";
                }
            }
        }
        
        // Velocity data
        outFile << "POINT_DATA " << nx * ny * nz << "\n";
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
        
        // Pressure data
        outFile << "SCALARS pressure float 1\n";
        outFile << "LOOKUP_TABLE default\n";
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    outFile << solver.getP(i, j, k) << "\n";
                }
            }
        }
        
        // Velocity magnitude
        outFile << "SCALARS velocity_magnitude float 1\n";
        outFile << "LOOKUP_TABLE default\n";
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    TNum u = solver.getU(i, j, k);
                    TNum v = solver.getV(i, j, k);
                    TNum w = solver.getW(i, j, k);
                    TNum mag = std::sqrt(u*u + v*v + w*w);
                    outFile << mag << "\n";
                }
            }
        }
        
        // Vorticity
        VectorObj<TNum> vort_x, vort_y, vort_z;
        solver.calculateVorticity(vort_x, vort_y, vort_z);
        
        outFile << "VECTORS vorticity float\n";
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    int idx = i + j * nx + k * nx * ny;
                    outFile << vort_x[idx] << " " << vort_y[idx] << " " << vort_z[idx] << "\n";
                }
            }
        }
        
        outFile.close();
    }
    
public:
    JetFlowSimulation(
        int nx_, int ny_, int nz_,
        TNum domainLength_, TNum domainWidth_, TNum domainHeight_,
        TNum jetVelocity_, TNum jetRadius_,
        TNum Re, TNum dt,
        std::string outputDir_ = "./output",
        int outputFrequency_ = 10
    ) : nx(nx_), ny(ny_), nz(nz_),
        domainLength(domainLength_), domainWidth(domainWidth_), domainHeight(domainHeight_),
        jetVelocity(jetVelocity_), jetRadius(jetRadius_),
        outputDir(outputDir_), outputFrequency(outputFrequency_),
        dx(domainLength_ / (nx_ - 1)), dy(domainWidth_ / (ny_ - 1)), dz(domainHeight_ / (nz_ - 1)),
        solver(nx_, ny_, nz_, dx, dy, dz, dt, Re) {
        
        createOutputDirectory();
        setupBoundaryConditions();
        setupInitialConditions();
    }
    
	void setupBoundaryConditions() {
        // Define jet inlet at the bottom center of the domain
        auto u_bc = [this](TNum x, TNum y, TNum z, TNum t) -> TNum {
            // Jet inlet at bottom (z=0)
            if (z < 1e-6) {
                // Calculate distance from center of inlet
                TNum centerX = domainLength / 2.0;
                TNum centerY = domainWidth / 2.0;
                TNum r = std::sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
                
                // Jet profile (parabolic)
                if (r <= jetRadius) {
                    return 0.0; // No x-velocity for jet
                }
            }
            
            // Pressure outlet at top (z=domainHeight)
            if (std::abs(z - domainHeight) < 1e-6) {
                // Zero gradient for velocity at pressure outlet
                return solver.getU(std::min(static_cast<int>(x/dx), nx-1), 
                                  std::min(static_cast<int>(y/dy), ny-1), 
                                  nz-2); // Use value from interior cell
            }
            
            return 0.0; // No-slip elsewhere
        };
        
        auto v_bc = [this](TNum x, TNum y, TNum z, TNum t) -> TNum {
            // Jet inlet at bottom (z=0)
            if (z < 1e-6) {
                // Calculate distance from center of inlet
                TNum centerX = domainLength / 2.0;
                TNum centerY = domainWidth / 2.0;
                TNum r = std::sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
                
                // Jet profile (parabolic)
                if (r <= jetRadius) {
                    return 0.0; // No y-velocity for jet
                }
            }
            
            // Pressure outlet at top (z=domainHeight)
            if (std::abs(z - domainHeight) < 1e-6) {
                // Zero gradient for velocity at pressure outlet
                return solver.getV(std::min(static_cast<int>(x/dx), nx-1), 
                                  std::min(static_cast<int>(y/dy), ny-1), 
                                  nz-2); // Use value from interior cell
            }
            
            return 0.0; // No-slip elsewhere
        };
        
        auto w_bc = [this](TNum x, TNum y, TNum z, TNum t) -> TNum {
            // Jet inlet at bottom (z=0)
            if (z < 1e-6) {
                // Calculate distance from center of inlet
                TNum centerX = domainLength / 2.0;
                TNum centerY = domainWidth / 2.0;
                TNum r = std::sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
                
                // Jet profile (parabolic)
                if (r <= jetRadius) {
                    return jetVelocity * (1.0 - (r / jetRadius) * (r / jetRadius));
                }
            }
            
            // Pressure outlet at top (z=domainHeight)
            if (std::abs(z - domainHeight) < 1e-6) {
                // For outflow, we use a zero gradient condition
                // This allows fluid to exit the domain naturally
                return solver.getW(std::min(static_cast<int>(x/dx), nx-1), 
                                  std::min(static_cast<int>(y/dy), ny-1), 
                                  nz-2); // Use value from interior cell
            }
            
            return 0.0; // No-slip elsewhere
        };
        
        // Set up pressure boundary conditions
        // This is handled implicitly in the pressure Poisson solver
        // by setting appropriate boundary conditions for velocity
        
        solver.setBoundaryConditions(u_bc, v_bc, w_bc);
        
        // Additional setup for pressure outlet
        // In the pressure projection step, we'll set p = 0 at the outlet
        // This is handled in the NavierStokesSolver3D class during the pressure solve
    }
    
    void setupInitialConditions() {
        // Initial conditions - fluid at rest
        auto u_init = [](TNum x, TNum y, TNum z) -> TNum { return 0.0; };
        auto v_init = [](TNum x, TNum y, TNum z) -> TNum { return 0.0; };
        auto w_init = [](TNum x, TNum y, TNum z) -> TNum { return 0.0; };
        
        solver.setInitialConditions(u_init, v_init, w_init);
    }
    
    void run(TNum duration) {
        int totalSteps = static_cast<int>(duration / solver.getDt());
        
        std::cout << "Starting jet flow simulation..." << std::endl;
        std::cout << "Domain: " << domainLength << " x " << domainWidth << " x " << domainHeight << std::endl;
        std::cout << "Grid: " << nx << " x " << ny << " x " << nz << std::endl;
        std::cout << "Reynolds number: " << solver.getRe() << std::endl;
        std::cout << "Jet velocity: " << jetVelocity << std::endl;
        std::cout << "Jet radius: " << jetRadius << std::endl;
        std::cout << "Time step: " << solver.getDt() << std::endl;
        std::cout << "Total steps: " << totalSteps << std::endl;
        
        // Initial output
        writeVTKFile(0, 0.0);
        
        for (int step = 1; step <= totalSteps; step++) {
            TNum time = step * solver.getDt();
            
            // Advance simulation
            solver.step(time);
            
            // Output results
            if (step % outputFrequency == 0) {
                std::cout << "Step " << step << "/" << totalSteps 
                          << ", Time: " << time 
                          << ", Kinetic Energy: " << solver.calculateKineticEnergy() << std::endl;
                
                writeVTKFile(step, time);
            }
        }
        
        std::cout << "Simulation completed." << std::endl;
    }
    
    // Accessor for the solver
    const NavierStokesSolver3D<TNum>& getSolver() const {
        return solver;
    }
};

// Example usage function
template <typename TNum = double>
void runJetFlowExample() {
    // Simulation parameters
    int nx = 50;
    int ny = 50;
    int nz = 100;
    TNum domainLength = 1.0;
    TNum domainWidth = 1.0;
    TNum domainHeight = 2.0;
    TNum jetVelocity = 10.0;
    TNum jetRadius = 0.2;
    TNum Re = 1000.0;  // Reynolds number
    TNum dt = 0.001;   // Time step
    TNum duration = 5.0; // Simulation duration
    
    // Create and run simulation
    JetFlowSimulation<TNum> simulation(
        nx, ny, nz,
        domainLength, domainWidth, domainHeight,
        jetVelocity, jetRadius,
        Re, dt,
        "jet_flow_output",
        5  
    );
    
    simulation.run(duration);
}

int main(){
	runJetFlowExample();
	return 0;
}
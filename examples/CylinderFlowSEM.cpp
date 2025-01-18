#include "SpectralElementMethod.hpp"
#include "SU2MeshAdapter.hpp"
#include "RungeKutta.hpp"
#include <iostream>
#include <fstream>
#include <cmath>

// Constants for the simulation
const double Reynolds = 100.0;    // Reynolds number for laminar vortex shedding
const double U_inlet = 1.0;       // Inlet velocity
const double dt = 0.001;          // Time step size
const size_t num_steps = 5000;    // Number of time steps
const size_t output_interval = 50; // Output solution every 50 steps
const double cylinder_radius = 0.5;// Cylinder radius
const double domain_length = 20.0; // Domain length
const double domain_height = 10.0; // Domain height

// Helper functions to access velocity components
double& u_comp(VectorObj<double>& v, size_t i) { return v[2*i]; }
double& v_comp(VectorObj<double>& v, size_t i) { return v[2*i + 1]; }
const double& u_comp(const VectorObj<double>& v, size_t i) { return v[2*i]; }
const double& v_comp(const VectorObj<double>& v, size_t i) { return v[2*i + 1]; }

// Define the Navier-Stokes operator
VectorObj<double> navierStokesOperator(const VectorObj<double>& velocity, SpectralElementMethod<double>& sem) {
    size_t total_dof = velocity.size() / 2; // Number of nodes
    VectorObj<double> result(velocity.size());
    
    // Assemble the diffusion term
    sem.assembleSystem();
    
    // Compute convection terms (u∂u/∂x + v∂u/∂y, u∂v/∂x + v∂v/∂y)
    VectorObj<double> convection(velocity.size());
    convection.zero();
    
    for (size_t e = 0; e < sem.num_elements; ++e) {
        auto& conn = sem.element_connectivity[e];
        for (size_t i = 0; i < conn.global_indices.size(); ++i) {
            size_t gi = conn.global_indices[i];
            double dudx = 0.0, dudy = 0.0, dvdx = 0.0, dvdy = 0.0;
            
            // Compute velocity gradients using spectral element basis functions
            for (size_t j = 0; j < conn.global_indices.size(); ++j) {
                size_t gj = conn.global_indices[j];
                double weight = sem.global_matrix(gi, gj);
                dudx += u_comp(velocity, gj) * weight;
                dudy += u_comp(velocity, gj) * weight;
                dvdx += v_comp(velocity, gj) * weight;
                dvdy += v_comp(velocity, gj) * weight;
            }
            
            // Compute convection terms
            u_comp(convection, gi) = u_comp(velocity, gi) * dudx + v_comp(velocity, gi) * dudy;
            v_comp(convection, gi) = u_comp(velocity, gi) * dvdx + v_comp(velocity, gi) * dvdy;
        }
    }
    
    // Combine terms: du/dt = -(u∂u/∂x + v∂u/∂y) + (1/Re)∇²u
    // Apply diffusion term to each component
    for (size_t i = 0; i < total_dof; ++i) {
        double diff_u = 0.0, diff_v = 0.0;
        for (size_t j = 0; j < total_dof; ++j) {
            double weight = sem.global_matrix(i, j);
            diff_u += weight * u_comp(velocity, j);
            diff_v += weight * v_comp(velocity, j);
        }
        
        u_comp(result, i) = -u_comp(convection, i) + diff_u / Reynolds;
        v_comp(result, i) = -v_comp(convection, i) + diff_v / Reynolds;
    }
    
    return result;
}

// Boundary conditions
bool isOnCylinder(const std::vector<double>& coords) {
    double x = coords[0];
    double y = coords[1];
    double r = std::sqrt(x*x + y*y);
    return std::abs(r - cylinder_radius) < 1e-6;
}

bool isAtInlet(const std::vector<double>& coords) {
    return coords[0] < 1e-6;
}

bool isAtOutlet(const std::vector<double>& coords) {
    return coords[0] > domain_length - 1e-6;
}

double inletVelocityX(const std::vector<double>& coords) {
    return U_inlet;
}

double inletVelocityY(const std::vector<double>& coords) {
    return 0.0;
}

int main() {
    // Define the mesh file
    std::string mesh_file = "../mesh/mesh_cylinder_lam.su2";
    
    // Initialize the Spectral Element Method
    size_t num_elements = 1000;  // Adjust based on mesh
    size_t polynomial_order = 4; // 4th order polynomials
    size_t num_dimensions = 2;   // 2D simulation
    std::vector<double> domain_bounds = {0.0, domain_length, 0.0, domain_height};
    
    auto dummy_operator = [](const std::vector<double>& coords) { return 0.0; };
    
    SpectralElementMethod<double> sem(
        num_elements,
        polynomial_order,
        num_dimensions,
        domain_bounds,
        mesh_file,
        dummy_operator,
        dummy_operator,
        dummy_operator
    );
    
    // Read the mesh
    SU2MeshAdapter mesh_adapter(mesh_file);
    mesh_adapter.readMesh(sem);
    
    // Initialize velocity field (2 components per node)
    size_t total_dof = sem.computeTotalDOF();
    VectorObj<double> velocity(2 * total_dof);
    velocity.zero();
    
    // Set initial conditions
    for (size_t i = 0; i < total_dof; ++i) {
        std::vector<double> coords = sem.getNodeCoordinates(i);
        
        if (isAtInlet(coords)) {
            u_comp(velocity, i) = inletVelocityX(coords);
            v_comp(velocity, i) = inletVelocityY(coords);
        } else if (isOnCylinder(coords)) {
            u_comp(velocity, i) = 0.0;
            v_comp(velocity, i) = 0.0;
        } else {
            u_comp(velocity, i) = U_inlet;
            v_comp(velocity, i) = 0.0;
        }
    }
    
    // Initialize Runge-Kutta solver
    RungeKutta<double, VectorObj<double>, VectorObj<double>> rk;
    
    // Define the system function for time stepping
    auto system_function = [&sem](const VectorObj<double>& v) {
        return navierStokesOperator(v, sem);
    };
    
    // Output files
    std::ofstream output_file("velocity_field.txt");
    
    // Time-stepping loop
    double time = 0.0;
    for (size_t step = 0; step < num_steps; ++step) {
        // Perform one Runge-Kutta step
        VectorObj<double> old_velocity = velocity;
        rk.solve(velocity, system_function, dt, 4); // 4th order RK
        
        // Apply boundary conditions
        for (size_t i = 0; i < total_dof; ++i) {
            std::vector<double> coords = sem.getNodeCoordinates(i);
            
            if (isAtInlet(coords)) {
                u_comp(velocity, i) = inletVelocityX(coords);
                v_comp(velocity, i) = inletVelocityY(coords);
            } else if (isOnCylinder(coords)) {
                u_comp(velocity, i) = 0.0;
                v_comp(velocity, i) = 0.0;
            }
        }
        
        // Update time
        time += dt;
        
        // Output the solution at specified intervals
        if (step % output_interval == 0) {
            output_file << "Time: " << time << std::endl;
            
            // Write node coordinates and velocity components
            for (size_t i = 0; i < total_dof; ++i) {
                std::vector<double> coords = sem.getNodeCoordinates(i);
                output_file << coords[0] << " " << coords[1] << " " 
                           << u_comp(velocity, i) << " " << v_comp(velocity, i) << std::endl;
            }
            
            // Compute and output maximum velocity magnitude
            double max_velocity = 0.0;
            for (size_t i = 0; i < total_dof; ++i) {
                double vel_mag = std::sqrt(
                    u_comp(velocity, i) * u_comp(velocity, i) + 
                    v_comp(velocity, i) * v_comp(velocity, i)
                );
                max_velocity = std::max(max_velocity, vel_mag);
            }
            
            std::cout << "Step: " << step << ", Time: " << time 
                      << ", Max Velocity: " << max_velocity << std::endl;
        }
    }
    
    output_file.close();
    return 0;
}
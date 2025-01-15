#ifndef SPECTRALELEMENTMETHOD_HPP
#define SPECTRALELEMENTMETHOD_HPP

#include <vector>
#include <array>
#include <functional>
#include "GaussianQuad.hpp"
#include "GMRES.hpp"
#include "DenseObj.hpp"
#include "VectorObj.hpp"
#include "SU2Mesh.hpp"

template<typename T, typename MatrixType = DenseObj<T>>
class SpectralElementMethod {
private:
    // Problem definition
    std::function<T(const std::vector<T>&)> pde_operator;
    std::function<T(const std::vector<T>&)> boundary_condition;
    std::function<T(const std::vector<T>&)> source_term;
    
    // Discretization parameters
    size_t num_elements;
    size_t polynomial_order;
    size_t num_dimensions;
    std::vector<T> domain_bounds;
    
    // Numerical components
    std::vector<Quadrature::GaussianQuadrature<T>> quadratures;
    MatrixType global_matrix;
    VectorObj<T> global_vector;
    
    // Element matrices
    std::vector<MatrixType> element_matrices;
    std::vector<VectorObj<T>> element_vectors;

    SU2Mesh mesh;

public:
    SpectralElementMethod(
        size_t elements,
        size_t order,
        size_t dims,
        const std::vector<T>& bounds,
        const std::string mesh_file,
        std::function<T(const std::vector<T>&)> op,
        std::function<T(const std::vector<T>&)> bc,
        std::function<T(const std::vector<T>&)> source
    ) : num_elements(elements),
        polynomial_order(order),
        num_dimensions(dims),
        domain_bounds(bounds),
        pde_operator(op),
        boundary_condition(bc),
        source_term(source) {
        
        initialize();
        initializeFromSU2Mesh(mesh_file);
    }

    void solve(VectorObj<T>& solution) {
        assembleSystem();
        
        // Solve using GMRES
        GMRES<T, MatrixType> solver;
        solver.solve(global_matrix, global_vector, solution, 1000, std::min(global_matrix.getRows(), global_matrix.getCols()), 1e-8);
    }

private:
    void initialize() {
        // Initialize quadrature rules
        for(size_t d = 0; d < num_dimensions; ++d) {
            quadratures.emplace_back(polynomial_order + 1);
        }
        
        // Initialize element matrices
        size_t dof_per_element = std::pow(polynomial_order + 1, num_dimensions);
        element_matrices.resize(num_elements, MatrixType(dof_per_element, dof_per_element));
        element_vectors.resize(num_elements, VectorObj<T>(dof_per_element));
        
        // Initialize global system
        size_t total_dof = num_elements * dof_per_element;
        global_matrix.resize(total_dof, total_dof);
        global_vector.resize(total_dof);
    }

    void initializeFromSU2Mesh(const std::string& filename) {
        // Load the SU2 mesh
        mesh.loadMesh(filename);
        mesh.printSummary();

        // Set dimensionality
        num_dimensions = mesh.dimensionality;

        // Set number of elements
        num_elements = mesh.elements.size();

        // Populate domain bounds (assumes a rectangular domain for simplicity)
        domain_bounds.resize(num_dimensions * 2);
        for (size_t d = 0; d < num_dimensions; ++d) {
            domain_bounds[d * 2] = std::numeric_limits<T>::max();
            domain_bounds[d * 2 + 1] = std::numeric_limits<T>::lowest();
            for (const auto& node : mesh.nodes) {
                domain_bounds[d * 2] = std::min(domain_bounds[d * 2], node.coordinates[d]);
                domain_bounds[d * 2 + 1] = std::max(domain_bounds[d * 2 + 1], node.coordinates[d]);
            }
        }

        // Initialize element matrices and vectors based on the SU2 mesh
        element_matrices.resize(num_elements);
        element_vectors.resize(num_elements);

        for (size_t i = 0; i < num_elements; ++i) {
            const auto& element = mesh.elements[i];
            size_t num_nodes_in_element = element.connectivity.size();

            // Example: Initialize element matrix and vector sizes
            element_matrices[i].resize(num_nodes_in_element, num_nodes_in_element);
            element_vectors[i].resize(num_nodes_in_element);
        }
    }

    void assembleSystem() {
        for(size_t e = 0; e < num_elements; ++e) {
            assembleElementMatrix(e);
            assembleElementVector(e);
            incorporateElement(e);
        }
        applyBoundaryConditions();
    }

    void assembleElementMatrix(size_t element_index) {
        auto& element_matrix = element_matrices[element_index];
        
        // Get element bounds
        std::vector<T> element_bounds = getElementBounds(element_index);
        
        // Compute basis function values and derivatives
        for(size_t i = 0; i < element_matrix.getRows(); ++i) {
            for(size_t j = 0; j < element_matrix.getCols(); ++j) {
                element_matrix(i,j) = computeElementIntegral(i, j, element_bounds);
            }
        }
    }

    void assembleElementVector(size_t element_index) {
        auto& element_vector = element_vectors[element_index];
        std::vector<T> element_bounds = getElementBounds(element_index);
        
        for(size_t i = 0; i < element_vector.size(); ++i) {
            element_vector[i] = computeSourceIntegral(i, element_bounds);
        }
    }

    T computeElementIntegral(size_t i, size_t j, const std::vector<T>& bounds) {
        T integral = 0;
        
        // Integrate over element using tensor product of quadrature rules
        for(const auto& quad : quadratures) {
            const auto& points = quad.getPoints();
            const auto& weights = quad.getWeights();
            
            for(size_t q = 0; q < points.size(); ++q) {
                std::vector<T> x = mapToPhysical(points[q], bounds);
                T basis_i = evaluateBasis(i, x);
                T basis_j = evaluateBasis(j, x);
                T jacobian = computeJacobian(bounds);
                
                integral += weights[q] * pde_operator(x) * basis_i * basis_j * jacobian;
            }
        }
        
        return integral;
    }

    T computeSourceIntegral(size_t i, const std::vector<T>& bounds) {
        T integral = 0;
        
        for(const auto& quad : quadratures) {
            const auto& points = quad.getPoints();
            const auto& weights = quad.getWeights();
            
            for(size_t q = 0; q < points.size(); ++q) {
                std::vector<T> x = mapToPhysical(points[q], bounds);
                T basis = evaluateBasis(i, x);
                T jacobian = computeJacobian(bounds);
                
                integral += weights[q] * source_term(x) * basis * jacobian;
            }
        }
        
        return integral;
    }

    void incorporateElement(size_t element_index) {
        size_t dof_per_element = std::pow(polynomial_order + 1, num_dimensions);
        size_t offset = element_index * dof_per_element;
        
        // Add element matrix to global matrix
        for(size_t i = 0; i < dof_per_element; ++i) {
            for(size_t j = 0; j < dof_per_element; ++j) {
                global_matrix(offset + i, offset + j) += element_matrices[element_index](i,j);
            }
            global_vector[offset + i] += element_vectors[element_index][i];
        }
    }

    void applyBoundaryConditions() {
        // Apply Dirichlet BCs
        for(size_t i = 0; i < global_matrix.getRows(); ++i) {
            if(isOnBoundary(i)) {
                // Set diagonal to 1 and off-diagonals to 0
                for(size_t j = 0; j < global_matrix.getCols(); ++j) {
                    global_matrix(i,j) = (i == j) ? 1.0 : 0.0;
                }
                global_vector[i] = boundary_condition({static_cast<T>(i)});
            }
        }
    }

    // Helper functions
    std::vector<T> getElementBounds(size_t element_index) {
        std::vector<T> bounds;
        for(size_t d = 0; d < num_dimensions; ++d) {
            T length = domain_bounds[2*d + 1] - domain_bounds[2*d];
            T dx = length / num_elements;
            bounds.push_back(domain_bounds[2*d] + element_index * dx);
            bounds.push_back(domain_bounds[2*d] + (element_index + 1) * dx);
        }
        return bounds;
    }

    std::vector<T> mapToPhysical(T xi, const std::vector<T>& bounds) {
        std::vector<T> x(num_dimensions);
        for(size_t d = 0; d < num_dimensions; ++d) {
            x[d] = bounds[2*d] + (bounds[2*d + 1] - bounds[2*d]) * (xi + 1) / 2;
        }
        return x;
    }

    T computeJacobian(const std::vector<T>& bounds) {
        T jacobian = 1.0;
        for(size_t d = 0; d < num_dimensions; ++d) {
            jacobian *= (bounds[2*d + 1] - bounds[2*d]) / 2;
        }
        return jacobian;
    }

    T evaluateBasis(size_t index, const std::vector<T>& x) {
        T result = 1.0;
        size_t nodes_per_dim = polynomial_order + 1;
        
        // Decompose global index into per-dimension indices
        std::vector<size_t> indices(num_dimensions);
        size_t remaining = index;
        for(size_t d = 0; d < num_dimensions; ++d) {
            indices[d] = remaining % nodes_per_dim;
            remaining /= nodes_per_dim;
        }
        
        // Compute basis as product of Legendre polynomials
        for(size_t d = 0; d < num_dimensions; ++d) {
            // Map x to [-1,1] interval
            T xi = 2.0 * (x[d] - domain_bounds[2*d]) / 
                  (domain_bounds[2*d + 1] - domain_bounds[2*d]) - 1.0;
            
            // Evaluate Legendre polynomial of order indices[d]
            if(indices[d] == 0) {
                result *= 1.0;
            } else if(indices[d] == 1) {
                result *= xi;
            } else {
                T p0 = 1.0;
                T p1 = xi;
                T pn;
                
                for(size_t n = 2; n <= indices[d]; ++n) {
                    pn = ((2*n - 1) * xi * p1 - (n - 1) * p0) / n;
                    p0 = p1;
                    p1 = pn;
                }
                result *= p1;
            }
        }
        
        return result;
    }


    bool isOnBoundary(size_t node_index) {
        size_t nodes_per_dim = polynomial_order + 1;
        size_t total_nodes_per_element = std::pow(nodes_per_dim, num_dimensions);
        size_t element_index = node_index / total_nodes_per_element;
        size_t local_index = node_index % total_nodes_per_element;

        // Check if element is on domain boundary
        if(element_index == 0 || element_index == num_elements - 1) {
            return true;
        }

        // Check if local node is on element boundary
        for(size_t d = 0; d < num_dimensions; ++d) {
            size_t pos = (local_index / static_cast<size_t>(std::pow(nodes_per_dim, d))) % nodes_per_dim;
            if(pos == 0 || pos == polynomial_order) {
                return true;
            }
        }
        return false;
    }
};

#endif // SPECTRALELEMENTMETHOD_HPP
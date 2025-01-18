#ifndef SPECTRALELEMENTMETHOD_HPP
#define SPECTRALELEMENTMETHOD_HPP

#include <vector>
#include <array>
#include <functional>
#include <cmath>
#include "GaussianQuad.hpp"
#include "GMRES.hpp"
#include "ConjugateGradient.hpp"
#include "DenseObj.hpp"
#include "VectorObj.hpp"

template<typename T, typename MatrixType = DenseObj<T>>
class SpectralElementMethod {
public:
    // Problem definition
    std::function<T(const std::vector<T>&)> pde_operator;
    std::function<T(const std::vector<T>&)> boundary_condition;
    std::function<T(const std::vector<T>&)> source_term;
  
    // Discretization parameters
    std::vector<size_t> num_elements_per_dim;
    size_t polynomial_order;
    size_t num_dimensions;
    std::vector<T> domain_bounds;

    struct ElementConnectivity {
        std::vector<size_t> global_indices;
        std::vector<std::vector<size_t>> face_nodes;
        std::vector<int> neighboring_elements;
    };
    std::vector<ElementConnectivity> element_connectivity;

    // Numerical components
    std::vector<Quadrature::GaussianQuadrature<T>> quadratures;

    MatrixType global_matrix;
    VectorObj<T> global_vector;
private:
    // Element matrices and basis functions
    std::vector<MatrixType> element_matrices;
    std::vector<VectorObj<T>> element_vectors;
    std::vector<std::vector<T>> basis_nodes; // Store GLL nodes for basis functions

    // Helper function for integer exponentiation
    size_t int_pow(size_t base, size_t exponent) {
        size_t result = 1;
        for (size_t i = 0; i < exponent; ++i) {
            result *= base;
        }
        return result;
    }

public:
    SpectralElementMethod(
        const std::vector<size_t>& elements_per_dim,
        size_t order,
        size_t dims,
        const std::vector<T>& bounds,
        const std::string mesh_file,
        std::function<T(const std::vector<T>&)> op,
        std::function<T(const std::vector<T>&)> bc,
        std::function<T(const std::vector<T>&)> source
    ) : num_elements_per_dim(elements_per_dim),
        polynomial_order(order),
        num_dimensions(dims),
        domain_bounds(bounds),
        pde_operator(op),
        boundary_condition(bc),
        source_term(source) {
        
        initialize();
    }

    void solve(VectorObj<T>& solution) {
        assembleSystem();
        
        // Solve using GMRES
        // GMRES<T, MatrixType, VectorObj<T>> solver;
        // solver.solve(global_matrix, global_vector, solution, 1000, std::min(global_matrix.getRows(), global_matrix.getCols()), 1e-8);

        // Solve using conjugate gradient
        ConjugateGrad<T, MatrixType, VectorObj<T>> solver(global_matrix, global_vector, 1000, 1e-8);
        solver.solve(solution);
    }

    void initialize() {
        // Initialize Gauss-Lobatto-Legendre quadrature rules
        for(size_t d = 0; d < num_dimensions; ++d) {
            quadratures.emplace_back(polynomial_order + 1);
        }
    
        // Initialize GLL nodes for basis functions
        initializeBasisNodes();
    
        // Initialize element matrices
        size_t dof_per_element = std::pow(polynomial_order + 1, num_dimensions);
        element_matrices.resize(total_elements(), MatrixType(dof_per_element, dof_per_element));
        element_vectors.resize(total_elements(), VectorObj<T>(dof_per_element));
    
        // Initialize global system
        size_t total_dof = computeTotalDOF();
        global_matrix.resize(total_dof, total_dof);
        global_vector.resize(total_dof);
    
        // Initialize connectivity information
        initializeConnectivity();
    }

    void initializeBasisNodes() {
        basis_nodes.resize(num_dimensions);
        for (size_t d = 0; d < num_dimensions; ++d) {
            basis_nodes[d].resize(polynomial_order + 1);
            for (size_t i = 0; i <= polynomial_order; ++i) {
                basis_nodes[d][i] = quadratures[d].getPoints()[i];
            }
        }
    }

    void initializeConnectivity() {
        element_connectivity.resize(total_elements());
        
        size_t nodes_per_dim = polynomial_order + 1;
        size_t dof_per_element = std::pow(nodes_per_dim, num_dimensions);
        
        for (size_t e = 0; e < total_elements(); ++e) {
            auto& conn = element_connectivity[e];
            conn.global_indices.resize(dof_per_element);
            
            // Compute element indices per dimension
            std::vector<size_t> elem_indices(num_dimensions);
            size_t temp = e;
            for (size_t d = 0; d < num_dimensions; ++d) {
                elem_indices[d] = temp / int_pow(num_elements_per_dim[d], num_dimensions - 1 - d);
                temp %= int_pow(num_elements_per_dim[d], num_dimensions - 1 - d);
            }
            
            // Compute global indices for each local node
            for (size_t i = 0; i < dof_per_element; ++i) {
                std::vector<size_t> local_indices = getLocalIndices(i);
                size_t global_index = 0;
                size_t stride = 1;
                for (size_t d = 0; d < num_dimensions; ++d) {
                    size_t global_pos = elem_indices[d] * nodes_per_dim + local_indices[d];
                    global_index += global_pos * stride;
                    stride *= (num_elements_per_dim[d] * nodes_per_dim);
                }
                conn.global_indices[i] = global_index;
            }
            
            // Initialize face nodes and neighboring elements
            initializeElementFaces(e);
        }
    }

    void initializeElementFaces(size_t element_index) {
        auto& conn = element_connectivity[element_index];
        size_t nodes_per_face = std::pow(polynomial_order + 1, num_dimensions - 1);
        conn.face_nodes.resize(2 * num_dimensions);
        conn.neighboring_elements.resize(2 * num_dimensions, -1);
        
        size_t nodes_per_dim = polynomial_order + 1;
        
        for (size_t d = 0; d < num_dimensions; ++d) {
            // Lower face in dimension d
            for (size_t i = 0; i < nodes_per_face; ++i) {
                std::vector<size_t> local_indices = getLocalIndices(i);
                local_indices.insert(local_indices.begin() + d, 0);
                conn.face_nodes[2*d].push_back(getLocalIndex(local_indices));
            }
            
            // Upper face in dimension d
            for (size_t i = 0; i < nodes_per_face; ++i) {
                std::vector<size_t> local_indices = getLocalIndices(i);
                local_indices.insert(local_indices.begin() + d, nodes_per_dim - 1);
                conn.face_nodes[2*d + 1].push_back(getLocalIndex(local_indices));
            }
            
            // Identify neighboring elements
            std::vector<size_t> elem_indices(num_dimensions);
            size_t temp = element_index;
            for (size_t dim = 0; dim < num_dimensions; ++dim) {
                elem_indices[dim] = temp / int_pow(num_elements_per_dim[dim], num_dimensions - 1 - dim);
                temp %= int_pow(num_elements_per_dim[dim], num_dimensions - 1 - dim);
            }
            
            if (elem_indices[d] > 0) {
                conn.neighboring_elements[2*d] = element_index - int_pow(num_elements_per_dim[d], d);
            }
            if (elem_indices[d] < num_elements_per_dim[d] - 1) {
                conn.neighboring_elements[2*d + 1] = element_index + int_pow(num_elements_per_dim[d], d);
            }
        }
    }

    std::vector<size_t> getLocalIndices(size_t linear_index) {
        std::vector<size_t> indices(num_dimensions);
        size_t temp = linear_index;
        size_t nodes_per_dim = polynomial_order + 1;
        
        for (size_t d = 0; d < num_dimensions; ++d) {
            indices[d] = temp % nodes_per_dim;
            temp /= nodes_per_dim;
        }
        
        return indices;
    }

    size_t getLocalIndex(const std::vector<size_t>& indices) {
        size_t index = 0;
        size_t nodes_per_dim = polynomial_order + 1;
        for (size_t d = 0; d < num_dimensions; ++d) {
            index += indices[d] * int_pow(nodes_per_dim, d);
        }
        return index;
    }

    T evaluateBasis(size_t basis_index, const std::vector<T>& xi) {
        std::vector<size_t> indices = getLocalIndices(basis_index);
        T result = 1.0;
        
        for (size_t d = 0; d < num_dimensions; ++d) {
            result *= evaluateLagrangePolynomial(indices[d], xi[d], basis_nodes[d]);
        }
        
        return result;
    }

    std::vector<T> evaluateBasisGradient(size_t basis_index, const std::vector<T>& xi) {
        std::vector<size_t> indices = getLocalIndices(basis_index);
        std::vector<T> gradient(num_dimensions);
        
        for (size_t d = 0; d < num_dimensions; ++d) {
            T prod = 1.0;
            for (size_t k = 0; k < num_dimensions; ++k) {
                if (k == d) {
                    prod *= evaluateLagrangePolynomialDerivative(indices[k], xi[k], basis_nodes[k]);
                } else {
                    prod *= evaluateLagrangePolynomial(indices[k], xi[k], basis_nodes[k]);
                }
            }
            gradient[d] = prod;
        }
        
        return gradient;
    }

    T evaluateLagrangePolynomial(size_t i, T x, const std::vector<T>& nodes) {
        T result = 1.0;
        for (size_t j = 0; j < nodes.size(); ++j) {
            if (j != i) {
                result *= (x - nodes[j]) / (nodes[i] - nodes[j]);
            }
        }
        return result;
    }

    T evaluateLagrangePolynomialDerivative(size_t i, T x, const std::vector<T>& nodes) {
        T sum = 0.0;
        for (size_t j = 0; j < nodes.size(); ++j) {
            if (j != i) {
                T prod = 1.0 / (nodes[i] - nodes[j]);
                for (size_t k = 0; k < nodes.size(); ++k) {
                    if (k != i && k != j) {
                        prod *= (x - nodes[k]) / (nodes[i] - nodes[k]);
                    }
                }
                sum += prod;
            }
        }
        return sum;
    }

    void assembleElementVector(size_t element_index) {
        auto& element_vector = element_vectors[element_index];
        element_vector.zero();
        
        std::vector<T> element_bounds = getElementBounds(element_index);
        
        // Get quadrature points and weights
        std::vector<std::vector<T>> quad_points(num_dimensions);
        std::vector<std::vector<T>> quad_weights(num_dimensions);
        for (size_t d = 0; d < num_dimensions; ++d) {
            quad_points[d] = quadratures[d].getPoints();
            quad_weights[d] = quadratures[d].getWeights();
        }
        
        // Nested quadrature loop
        std::vector<size_t> quad_indices(num_dimensions, 0);
        bool done = false;
        
        while (!done) {
            // Compute quadrature point coordinates and weights
            T weight = 1.0;
            std::vector<T> quad_coords(num_dimensions);
            
            for (size_t d = 0; d < num_dimensions; ++d) {
                quad_coords[d] = quad_points[d][quad_indices[d]];
                weight *= quad_weights[d][quad_indices[d]];
            }
            
            // Map to physical space
            std::vector<T> phys_coords = mapToPhysical(quad_coords, element_bounds);
            
            // Evaluate basis functions at quadrature point
            std::vector<T> basis_values = computeBasisFunctions(quad_coords);
            
            // Compute Jacobian determinant
            T jacobian_det = computeJacobianDeterminant(element_bounds);
            
            // Evaluate source term
            T source_value = source_term(phys_coords);
            
            // Add contribution to element vector
            for (size_t i = 0; i < basis_values.size(); ++i) {
                element_vector[i] += weight * source_value * basis_values[i] * jacobian_det;
            }
            
            // Update quadrature indices
            for (int d = num_dimensions - 1; d >= 0; --d) {
                quad_indices[d]++;
                if (quad_indices[d] < quad_points[d].size()) {
                    break;
                }
                quad_indices[d] = 0;
                if (d == 0) {
                    done = true;
                }
            }
        }
    }

    void incorporateElement(size_t element_index) {
        const auto& conn = element_connectivity[element_index];
        
        // Add element matrix contributions to global matrix
        for (size_t i = 0; i < conn.global_indices.size(); ++i) {
            size_t gi = conn.global_indices[i];
            for (size_t j = 0; j < conn.global_indices.size(); ++j) {
                size_t gj = conn.global_indices[j];
                global_matrix(gi, gj) += element_matrices[element_index](i, j);
            }
            global_vector[gi] += element_vectors[element_index][i];
        }
    }

    std::vector<T> mapToPhysical(const std::vector<T>& xi, const std::vector<T>& bounds) {
        std::vector<T> x(num_dimensions);
        for (size_t d = 0; d < num_dimensions; ++d) {
            x[d] = 0.5 * ((1.0 - xi[d]) * bounds[2*d] + (1.0 + xi[d]) * bounds[2*d + 1]);
        }
        return x;
    }

    T computeJacobianDeterminant(const std::vector<T>& bounds) {
        T det = 1.0;
        for (size_t d = 0; d < num_dimensions; ++d) {
            det *= 0.5 * (bounds[2*d + 1] - bounds[2*d]);
        }
        return det;
    }

    void applyBoundaryConditions() {
        // Apply Dirichlet boundary conditions
        size_t total_dof = computeTotalDOF();
        
        for (size_t i = 0; i < total_dof; ++i) {
            std::vector<T> node_coords = getNodeCoordinates(i);
            
            if (isOnBoundary(node_coords)) {
                // Set row to zero except diagonal
                for (size_t j = 0; j < total_dof; ++j) {
                    global_matrix(i, j) = (i == j) ? 1.0 : 0.0;
                }
                
                // Set right-hand side to boundary value
                global_vector[i] = boundary_condition(node_coords);
            }
        }
    }

    bool isOnBoundary(const std::vector<T>& coords) {
        for (size_t d = 0; d < num_dimensions; ++d) {
            if (std::abs(coords[d] - domain_bounds[2*d]) < 1e-10 ||
                std::abs(coords[d] - domain_bounds[2*d + 1]) < 1e-10) {
                return true;
            }
        }
        return false;
    }

    std::vector<T> getNodeCoordinates(size_t global_index) {
        // Convert global index to element and local indices
        std::vector<size_t> elem_indices(num_dimensions);
        size_t temp = global_index;
        size_t stride = 1;
        size_t nodes_per_dim = polynomial_order + 1;
        
        for (size_t d = 0; d < num_dimensions; ++d) {
            elem_indices[d] = temp / stride / nodes_per_dim;
            temp -= elem_indices[d] * stride * nodes_per_dim;
            stride *= num_elements_per_dim[d] * nodes_per_dim;
        }
        
        // Compute physical coordinates
        std::vector<T> coords(num_dimensions);
        for (size_t d = 0; d < num_dimensions; ++d) {
            coords[d] = domain_bounds[2*d] + 
                (domain_bounds[2*d + 1] - domain_bounds[2*d]) * 
                (static_cast<T>(elem_indices[d]) + 0.5) / num_elements_per_dim[d];
        }
        return coords;
    }

    size_t computeTotalDOF() {
        size_t total_dof = 1;
        size_t nodes_per_dim = polynomial_order + 1;
        for (size_t d = 0; d < num_dimensions; ++d) {
            total_dof *= num_elements_per_dim[d] * nodes_per_dim;
        }
        return total_dof;
    }

    void assembleSystem() {
        // Assemble element matrices and vectors
        for (size_t e = 0; e < total_elements(); ++e) {
            assembleElementMatrix(e);
            assembleElementVector(e);
            incorporateElement(e);
        }
        
        // Apply boundary conditions
        applyBoundaryConditions();
    }

    void assembleElementMatrix(size_t element_index) {
        auto& element_matrix = element_matrices[element_index];
        element_matrix.zero();
        
        std::vector<T> element_bounds = getElementBounds(element_index);
        
        // Get quadrature points and weights
        std::vector<std::vector<T>> quad_points(num_dimensions);
        std::vector<std::vector<T>> quad_weights(num_dimensions);
        for (size_t d = 0; d < num_dimensions; ++d) {
            quad_points[d] = quadratures[d].getPoints();
            quad_weights[d] = quadratures[d].getWeights();
        }
        
        // Nested quadrature loop
        std::vector<size_t> quad_indices(num_dimensions, 0);
        bool done = false;
        
        while (!done) {
            // Compute quadrature point coordinates and weights
            T weight = 1.0;
            std::vector<T> quad_coords(num_dimensions);
            
            for (size_t d = 0; d < num_dimensions; ++d) {
                quad_coords[d] = quad_points[d][quad_indices[d]];
                weight *= quad_weights[d][quad_indices[d]];
            }
            
            // Map to physical space
            std::vector<T> phys_coords = mapToPhysical(quad_coords, element_bounds);
            
            // Evaluate basis functions and their gradients at quadrature point
            std::vector<T> basis_values = computeBasisFunctions(quad_coords);
            std::vector<std::vector<T>> basis_gradients(basis_values.size());
            for (size_t i = 0; i < basis_values.size(); ++i) {
                basis_gradients[i] = evaluateBasisGradient(i, quad_coords);
            }
            
            // Compute Jacobian determinant
            T jacobian_det = computeJacobianDeterminant(element_bounds);
            
            // Compute the element matrix contribution
            for (size_t i = 0; i < basis_values.size(); ++i) {
                for (size_t j = 0; j < basis_values.size(); ++j) {
                    T stiffness = 0.0;
                    for (size_t d = 0; d < num_dimensions; ++d) {
                        stiffness += basis_gradients[i][d] * basis_gradients[j][d];
                    }
                    element_matrix(i, j) += weight * stiffness * jacobian_det;
                }
            }
            
            // Update quadrature indices
            for (int d = num_dimensions - 1; d >= 0; --d) {
                quad_indices[d]++;
                if (quad_indices[d] < quad_points[d].size()) {
                    break;
                }
                quad_indices[d] = 0;
                if (d == 0) {
                    done = true;
                }
            }
        }
    }

    std::vector<T> computeBasisFunctions(const std::vector<T>& xi) {
        std::vector<T> basis_values(std::pow(polynomial_order + 1, num_dimensions));
        for (size_t i = 0; i < basis_values.size(); ++i) {
            basis_values[i] = evaluateBasis(i, xi);
        }
        return basis_values;
    }

    std::vector<T> getElementBounds(size_t element_index) {
        std::vector<T> bounds(2 * num_dimensions);
        std::vector<size_t> elem_indices(num_dimensions);
        size_t temp = element_index;
        for (size_t d = 0; d < num_dimensions; ++d) {
            elem_indices[d] = temp / int_pow(num_elements_per_dim[d], num_dimensions - 1 - d);
            temp %= int_pow(num_elements_per_dim[d], num_dimensions - 1 - d);
        }
        
        for (size_t d = 0; d < num_dimensions; ++d) {
            bounds[2*d] = domain_bounds[2*d] + 
                (domain_bounds[2*d + 1] - domain_bounds[2*d]) * 
                elem_indices[d] / num_elements_per_dim[d];
            bounds[2*d + 1] = domain_bounds[2*d] + 
                (domain_bounds[2*d + 1] - domain_bounds[2*d]) * 
                (elem_indices[d] + 1) / num_elements_per_dim[d];
        }
        return bounds;
    }

    size_t total_elements() {
        size_t total = 1;
        for (size_t d = 0; d < num_dimensions; ++d) {
            total *= num_elements_per_dim[d];
        }
        return total;
    }
};

#endif // SPECTRALELEMENTMETHOD_HPP
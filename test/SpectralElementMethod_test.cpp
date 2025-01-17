#include "SpectralElementMethod.hpp"
#include "gtest/gtest.h"
#include <cmath>

// Test fixture for SpectralElementMethod
class SpectralElementMethodTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Define the problem parameters
        num_elements = 1;
        polynomial_order = 30;
        num_dimensions = 1;
        domain_bounds = {0.0, 1.0};

        // Define the PDE operator, boundary condition, and source term
        pde_operator = [](const std::vector<double>& x) -> double {
            return -1.0; // -d^2u/dx^2
        };

        boundary_condition = [](const std::vector<double>& x) -> double {
            return 0.0; // Dirichlet boundary condition
        };

        source_term = [](const std::vector<double>& x) -> double {
            return 1.0; // Source term f(x) = 1
        };

        // Create the Spectral Element Method solver
        sem = new SpectralElementMethod<double>(
            num_elements,
            polynomial_order,
            num_dimensions,
            domain_bounds,
            "", // No mesh file
            pde_operator,
            boundary_condition,
            source_term
        );
    }

    void TearDown() override {
        delete sem;
    }

    size_t num_elements;
    size_t polynomial_order;
    size_t num_dimensions;
    std::vector<double> domain_bounds;
    std::function<double(const std::vector<double>&)> pde_operator;
    std::function<double(const std::vector<double>&)> boundary_condition;
    std::function<double(const std::vector<double>&)> source_term;
    SpectralElementMethod<double>* sem;
};

// Test case for solving the Poisson equation
TEST_F(SpectralElementMethodTest, SolvesPoissonEquation) {
    VectorObj<double> solution;
    sem->solve(solution);

    // Check the size of the solution vector
    EXPECT_EQ(solution.size(), num_elements * pow(polynomial_order + 1, num_dimensions));

    // Check the boundary conditions
    EXPECT_NEAR(solution[0], 0.0, 1e-10);
    EXPECT_NEAR(solution[solution.size() - 1], 0.0, 1e-10);

    // Check the solution at some interior points
    for (size_t i = 1; i < solution.size() - 1; ++i) {
        double x = static_cast<double>(i) / (num_elements * polynomial_order);
        double exact_solution = 0.5 * x * (1-x); // Exact solution for f(x) = 1
        EXPECT_NEAR(solution[i], exact_solution, 1e-4);
    }
}

// Test case for checking the assembly of the global matrix
TEST_F(SpectralElementMethodTest, AssemblesGlobalMatrix) {
    sem->assembleSystem();

    // Check the size of the global matrix
    size_t total_dof = sem->computeTotalDOF();
    EXPECT_EQ(sem->global_matrix.getRows(), total_dof);
    EXPECT_EQ(sem->global_matrix.getCols(), total_dof);

    for(size_t i = 0; i<sem->global_matrix.getRows(); i++){
        for(size_t j = 0; j<sem->global_matrix.getCols(); j++){
            printf("%0.4f ", sem->global_matrix(i,j));
        }
        std::cout << std::endl;
    }

    // Check that the global matrix is symmetric
    for (size_t i = 0; i < total_dof; ++i) {
        for (size_t j = 0; j < total_dof; ++j) {
            EXPECT_NEAR(sem->global_matrix(i, j), sem->global_matrix(j, i), 1e-10);
        }
    }
}

// Test case for checking the application of boundary conditions
TEST_F(SpectralElementMethodTest, AppliesBoundaryConditions) {
    sem->assembleSystem();
    sem->applyBoundaryConditions();

    // Check that the boundary conditions are applied correctly
    size_t total_dof = sem->computeTotalDOF();
    for (size_t i = 0; i < total_dof; ++i) {
        std::vector<double> node_coords = sem->getNodeCoordinates(i);
        if (sem->isOnBoundary(node_coords)) {
            for (size_t j = 0; j < total_dof; ++j) {
                if (i == j) {
                    EXPECT_NEAR(sem->global_matrix(i, j), 1.0, 1e-10);
                } else {
                    EXPECT_NEAR(sem->global_matrix(i, j), 0.0, 1e-10);
                }
            }
            EXPECT_NEAR(sem->global_vector[i], 0.0, 1e-10);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
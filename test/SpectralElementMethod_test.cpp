#include <gtest/gtest.h>
#include "SpectralElementMethod.hpp"
#include <vector>
#include <functional>

// Test fixture for SpectralElementMethod
class SpectralElementMethodTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Define mock functions
        pde_operator = [](const std::vector<double>& x) -> double {
            return 0.0; // Mock PDE operator
        };

        boundary_condition = [](const std::vector<double>& x) -> double {
            return 0.0; // Mock boundary condition
        };

        source_term = [](const std::vector<double>& x) -> double {
            return 0.0; // Mock source term
        };

        // Define discretization parameters
        num_elements_per_dim = {2, 2}; // 2x2 elements in 2D
        polynomial_order = 2;
        num_dimensions = 2;
        domain_bounds = {0.0, 1.0, 0.0, 1.0}; // Domain [0,1]x[0,1]
    }

    std::function<double(const std::vector<double>&)> pde_operator;
    std::function<double(const std::vector<double>&)> boundary_condition;
    std::function<double(const std::vector<double>&)> source_term;
    std::vector<size_t> num_elements_per_dim;
    size_t polynomial_order;
    size_t num_dimensions;
    std::vector<double> domain_bounds;
};

// Test initialization of the SpectralElementMethod class
TEST_F(SpectralElementMethodTest, Initialization) {
    SpectralElementMethod<double, DenseObj<double>> sem(
        num_elements_per_dim,
        polynomial_order,
        num_dimensions,
        domain_bounds,
        "", // No mesh file
        pde_operator,
        boundary_condition,
        source_term
    );

    // Check if the total number of elements is correct
    EXPECT_EQ(sem.total_elements(), 4);

    // Check if the total degrees of freedom is correct
    size_t expected_dof = (num_elements_per_dim[0] * (polynomial_order + 1)) *
                          (num_elements_per_dim[1] * (polynomial_order + 1));
    EXPECT_EQ(sem.computeTotalDOF(), expected_dof);
}

// Test assembly of the global system
TEST_F(SpectralElementMethodTest, Assembly) {
    SpectralElementMethod<double, DenseObj<double>> sem(
        num_elements_per_dim,
        polynomial_order,
        num_dimensions,
        domain_bounds,
        "", // No mesh file
        pde_operator,
        boundary_condition,
        source_term
    );

    sem.assembleSystem();

    // Check if the global matrix and vector are non-empty
    EXPECT_GT(sem.global_matrix.getRows(), 0);
    EXPECT_GT(sem.global_matrix.getCols(), 0);
    EXPECT_GT(sem.global_vector.size(), 0);
}

// Test solving the system
TEST_F(SpectralElementMethodTest, Solve) {
    SpectralElementMethod<double, DenseObj<double>> sem(
        num_elements_per_dim,
        polynomial_order,
        num_dimensions,
        domain_bounds,
        "", // No mesh file
        pde_operator,
        boundary_condition,
        source_term
    );

    sem.assembleSystem();

    VectorObj<double> solution(sem.computeTotalDOF());
    sem.solve(solution);

    for(size_t i = 0; i < solution.size(); i++){
        printf("%0.4f ", solution[i]);
    }
    std::cout << std::endl;

    // Check if the solution vector is initialized
    EXPECT_EQ(solution.size(), sem.computeTotalDOF());
}

// Test application of boundary conditions
TEST_F(SpectralElementMethodTest, BoundaryConditions) {
    SpectralElementMethod<double, DenseObj<double>> sem(
        num_elements_per_dim,
        polynomial_order,
        num_dimensions,
        domain_bounds,
        "", // No mesh file
        pde_operator,
        boundary_condition,
        source_term
    );

    sem.assembleSystem();
    sem.applyBoundaryConditions();

    // Check if boundary nodes have the boundary values
    for (size_t i = 0; i < sem.computeTotalDOF(); ++i) {
        std::vector<double> coords = sem.getNodeCoordinates(i);
        if (sem.isOnBoundary(coords)) {
            EXPECT_EQ(sem.global_vector[i], boundary_condition(coords));
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
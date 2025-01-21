#include "../src/PDEs/SpectralElemMethod.hpp"
#include "gtest/gtest.h"

TEST(SpectralElementMethodTest, AssembleSystem) {
    // Setup test parameters
    std::vector<size_t> elements = {2, 2};
    size_t order = 3;
    size_t dimensions = 2;
    std::vector<double> bounds = {0.0, 1.0, 0.0, 1.0};
    auto pde_op = [](const std::vector<double>&) { return 0.0; };
    auto boundary_cond = [](const std::vector<double>&) { return 0.0; };
    auto source = [](const std::vector<double>&) { return 1.0; };

    SpectralElementMethod<double> sem(elements, order, dimensions, bounds, pde_op, boundary_cond, source);
    sem.assembleSystem();

    // Add assertions to verify the system matrix is correctly assembled
    // Example: EXPECT_EQ(expected_value, actual_value);
}

TEST(SpectralElementMethodTest, Solve) {
    // Setup test parameters
    std::vector<size_t> elements = {2, 2};
    size_t order = 3;
    size_t dimensions = 2;
    std::vector<double> bounds = {0.0, 1.0, 0.0, 1.0};
    auto pde_op = [](const std::vector<double>&) { return 0.0; };
    auto boundary_cond = [](const std::vector<double>&) { return 0.0; };
    auto source = [](const std::vector<double>&) { return 1.0; };

    SpectralElementMethod<double> sem(elements, order, dimensions, bounds, pde_op, boundary_cond, source);
    VectorObj<double> solution(4);
    sem.solve(solution);

    // Add assertions to verify the solution is correct
    // Example: EXPECT_NEAR(expected_value, solution[i], tolerance);
}
#include <gtest/gtest.h>
#include "NewtonMethod.hpp"
#include "VectorObj.hpp"
#include "DenseObj.hpp"

// Define the nonlinear function F(x)
VectorObj<double> F(const VectorObj<double> &x) {
    VectorObj<double> result(2);
    result[0] = x[0] * x[0] + x[1] * x[1] - 1; // x^2 + y^2 - 1 = 0
    result[1] = x[0] - x[1];                  // x - y = 0
    return result;
}

// Define the Jacobian function J(x)
DenseObj<double> J(const VectorObj<double> &x) {
    DenseObj<double> J_x(2, 2);
    J_x(0, 0) = 2 * x[0]; // dF1/dx = 2x
    J_x(0, 1) = 2 * x[1]; // dF1/dy = 2y
    J_x(1, 0) = 1;        // dF2/dx = 1
    J_x(1, 1) = -1;       // dF2/dy = -1
    return J_x;
}

// Test fixture for NewtonMethod
class NewtonMethodTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initial guess
        x = VectorObj<double>(2);
        x[0] = 1.0;
        x[1] = 1.0;

        // Create NewtonMethod object
        newton = NewtonMethod<double, DenseObj<double>>(1e-6, 1000, 1.0);
    }

    VectorObj<double> x;
   	NewtonMethod<double, DenseObj<double>> newton;
};

// Test case: Check if NewtonMethod converges to the correct solution
TEST_F(NewtonMethodTest, ConvergesToCorrectSolution) {
    // Solve the system
    newton.solve(x, F, J);

    // Expected solution
    double expected_solution = 1.0 / std::sqrt(2.0); // 1/sqrt(2) ≈ 0.707107

    // Check if the solution is correct within a small tolerance
    EXPECT_NEAR(x[0], expected_solution, 1e-6);
    EXPECT_NEAR(x[1], expected_solution, 1e-6);
}

// Test case: Check if NewtonMethod converges within the maximum number of iterations
TEST_F(NewtonMethodTest, ConvergesWithinMaxIterations) {
    // Solve the system
    newton.solve(x, F, J);

    // Check if the residual is below the tolerance
    VectorObj<double> residual = F(x);
    double residual_norm = residual.L2norm();
    EXPECT_LT(residual_norm, 1e-6);
}

// Test case: Check if NewtonMethod handles a singular Jacobian gracefully
TEST_F(NewtonMethodTest, HandlesSingularJacobian) {
    // Modify the Jacobian to make it singular
    auto singular_J = [](const VectorObj<double> &x) {
        DenseObj<double> J_x(2, 2);
        J_x(0, 0) = 0.0; // Singular Jacobian
        J_x(0, 1) = 0.0;
        J_x(1, 0) = 0.0;
        J_x(1, 1) = 0.0;
        return J_x;
    };

    // Attempt to solve the system with a singular Jacobian
    EXPECT_THROW(newton.solve(x, F, singular_J), std::runtime_error);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
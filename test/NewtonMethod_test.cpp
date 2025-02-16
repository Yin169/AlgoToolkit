#include <gtest/gtest.h>
#include "NewtonMethod.hpp"
#include "VectorObj.hpp"
#include "DenseObj.hpp"
#include <cmath>
#include <vector>

// Define a larger nonlinear system F(x) with 5 equations
VectorObj<double> F(const VectorObj<double>& x) {
    VectorObj<double> result(5, 0.0);
    // System of equations:
    // x1^2 + x2^2 + x3^2 + x4^2 + x5^2 = 1
    // x1 - x2 + x3 = 0
    // x2 - x3 + x4 = 0
    // x3 - x4 + x5 = 0
    // x1*x4 - x2*x3 + x5 = 0
    result[0] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3] + x[4]*x[4] - 1.0;
    result[1] = x[0] - x[1] + x[2];
    result[2] = x[1] - x[2] + x[3];
    result[3] = x[2] - x[3] + x[4];
    result[4] = x[0]*x[3] - x[1]*x[2] + x[4];
    return result;
}

// Define the Jacobian matrix for the larger system
DenseObj<double> J(const VectorObj<double>& x) {
    DenseObj<double> J_x(5, 5);
    
    // First row: partial derivatives of first equation
    J_x(0, 0) = 2.0 * x[0];
    J_x(0, 1) = 2.0 * x[1];
    J_x(0, 2) = 2.0 * x[2];
    J_x(0, 3) = 2.0 * x[3];
    J_x(0, 4) = 2.0 * x[4];
    
    // Second row: partial derivatives of second equation
    J_x(1, 0) = 1.0;
    J_x(1, 1) = -1.0;
    J_x(1, 2) = 1.0;
    J_x(1, 3) = 0.0;
    J_x(1, 4) = 0.0;
    
    // Third row: partial derivatives of third equation
    J_x(2, 0) = 0.0;
    J_x(2, 1) = 1.0;
    J_x(2, 2) = -1.0;
    J_x(2, 3) = 1.0;
    J_x(2, 4) = 0.0;
    
    // Fourth row: partial derivatives of fourth equation
    J_x(3, 0) = 0.0;
    J_x(3, 1) = 0.0;
    J_x(3, 2) = 1.0;
    J_x(3, 3) = -1.0;
    J_x(3, 4) = 1.0;
    
    // Fifth row: partial derivatives of fifth equation
    J_x(4, 0) = x[3];
    J_x(4, 1) = -x[2];
    J_x(4, 2) = -x[1];
    J_x(4, 3) = x[0];
    J_x(4, 4) = 1.0;
    
    return J_x;
}

class NewtonMethodLargeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initial guess vector with explicit size and initial value
        x = VectorObj<double>(5, 0.5);

        // Create NewtonMethod object with stricter tolerance
        newton = NewtonMethod<double, DenseObj<double>>(1e-10, 2000, 0.8);
    }

    VectorObj<double> x;
    NewtonMethod<double, DenseObj<double>> newton;
};

// Test case: Verify convergence for the larger system
TEST_F(NewtonMethodLargeTest, ConvergesForLargeSystem) {
    // Solve the system
    newton.solve(x, F, J);

    // Check if the residual is sufficiently small
    VectorObj<double> residual = F(x);
    double residual_norm = residual.L2norm();
    EXPECT_LT(residual_norm, 1e-8);

    // Verify that the solution satisfies the first equation (sum of squares = 1)
    double sum_squares = x * x; // Using VectorObj's dot product operator
    EXPECT_NEAR(sum_squares, 1.0, 1e-7);
}

// Test case: Test vector operations used in Newton method
TEST_F(NewtonMethodLargeTest, VectorOperationsWork) {
    VectorObj<double> v1(5, 1.0);
    VectorObj<double> v2(5, 2.0);
    
    // Test vector addition
    VectorObj<double> sum = v1 + v2;
    for(int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(sum[i], 3.0);
    }
    
    // Test scalar multiplication
    VectorObj<double> scaled = v1 * 2.0;
    for(int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(scaled[i], 2.0);
    }
    
    // Test dot product
    double dot = v1 * v2;
    EXPECT_DOUBLE_EQ(dot, 10.0);
    
    // Test L2 norm
    double norm = v1.L2norm();
    EXPECT_NEAR(norm, std::sqrt(5.0), 1e-10);
}

// Test case: Check convergence with different initial guesses
TEST_F(NewtonMethodLargeTest, ConvergesWithDifferentInitialGuesses) {
    std::vector<std::vector<double>> initial_values = {
        {0.2, 0.2, 0.2, 0.2, 0.2},
        {0.8, 0.8, 0.8, 0.8, 0.8},
        {0.1, 0.3, 0.5, 0.7, 0.9}
    };

    for (const auto& values : initial_values) {
        x = VectorObj<double>(5);
        for(int i = 0; i < 5; ++i) {
            x[i] = values[i];
        }
        
        newton.solve(x, F, J);
        
        VectorObj<double> residual = F(x);
        double residual_norm = residual.L2norm();
        EXPECT_LT(residual_norm, 1e-8);
    }
}

// Test case: Check vector resizing and zero initialization
TEST_F(NewtonMethodLargeTest, HandlesDynamicVectorOperations) {
    x.resize(10);
    EXPECT_EQ(x.size(), 10);
    
    x.zero();
    for(size_t i = 0; i < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(x[i], 0.0);
    }
    
    // Resize back to original size for Newton method
    x.resize(5);
    for(int i = 0; i < 5; ++i) {
        x[i] = 0.5;
    }
    
    newton.solve(x, F, J);
    VectorObj<double> residual = F(x);
    EXPECT_LT(residual.L2norm(), 1e-8);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#include "gtest/gtest.h"
#include "GMRES.hpp"
#include "SparseObj.hpp"
#include "DenseObj.hpp"
#include "VectorObj.hpp"
#include <cmath>

class GMRESTest : public ::testing::Test {
protected:
    void SetUp() override {
        int n = 3;
        // Create a simple 3x3 test matrix
        matrix = DenseObj<double>(n, n);
        matrix(0, 0) =  4.0;
        matrix(0, 1) =  -1.0;
        matrix(0, 2) = 0.0;
        matrix(1, 0) = -1.0;
        matrix(1, 1) = 4.0;
        matrix(1, 2) =  -1.0;
        matrix(2, 0) = 0.0;
        matrix(2, 1) = -1.0;
        matrix(2, 2) = 4.0;
        // matrix.finalize();

        // Create right-hand side vector b = [1, 5, 0]
        b = VectorObj<double>(n);
        b[0] = 1.0;
        b[1] = 5.0;
        b[2] = 0.0;

        // Initial guess x0 = [0, 0, 0]
        x = VectorObj<double>(3, 0.0);

        solver = GMRES<double, DenseObj<double>>();
    }

    DenseObj<double> matrix;
    VectorObj<double> b;
    VectorObj<double> x;
    GMRES<double, DenseObj<double>> solver;
};

TEST_F(GMRESTest, BasicSolverTest) {
    double tol = 1e-12;
    int maxIter = 10000;
    int krylovDim = 3;
    
    solver.solve(matrix, b, x, maxIter, krylovDim, tol);

    EXPECT_NEAR(x[0], 0.625, 1e-5);
    EXPECT_NEAR(x[1], 1.5, 1e-5);
    EXPECT_NEAR(x[2], 0.375, 1e-5);
}

TEST_F(GMRESTest, ZeroRHSTest) {
    VectorObj<double> zero_b(3, 0.0);
    VectorObj<double> zero_x(3, 0.0);
    
    solver.solve(matrix, zero_b, zero_x, 10000, 3, 1e-12);

    // Solution should be zero vector
    EXPECT_NEAR(zero_x[0], 0.0, 1e-10);
    EXPECT_NEAR(zero_x[1], 0.0, 1e-10);
    EXPECT_NEAR(zero_x[2], 0.0, 1e-10);
}

TEST_F(GMRESTest, ConvergenceTest) {
    VectorObj<double> x_initial = x;
    solver.solve(matrix, b, x, 10000, 3, 1e-12);
    
    // Verify residual
    VectorObj<double> residual = b - matrix * x;
    double residual_norm = residual.L2norm();
    EXPECT_LT(residual_norm, 1e-6);
}

TEST_F(GMRESTest, DimensionMismatchTest) {
    VectorObj<double> wrong_size_x(4, 0.0);
    EXPECT_THROW(solver.solve(matrix, b, wrong_size_x, 10000, 3, 1e-12), std::invalid_argument);
}

TEST_F(GMRESTest, KrylovDimensionTest) {
    // Test with different Krylov subspace dimensions
    std::vector<int> krylov_dims = {1, 2, 3};
    for (int dim : krylov_dims) {
        VectorObj<double> x_test = x;
        solver.solve(matrix, b, x_test, 10000, dim, 1e-12);
        
        VectorObj<double> residual = b - matrix * x_test;
        EXPECT_LT(residual.L2norm(), 1e-6);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
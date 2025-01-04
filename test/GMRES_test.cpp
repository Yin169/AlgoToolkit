#include "gtest/gtest.h"
#include "GMRES.hpp"
#include "SparseObj.hpp"
#include "VectorObj.hpp"
#include <cmath>

class GMRESTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple 3x3 test matrix
        matrix = SparseMatrixCSC<double>(4, 4);
        matrix.addValue(0, 0, 4.0);
        matrix.addValue(1, 1, 3.0);
        matrix.addValue(2, 2, 2.0);
        matrix.addValue(3, 3, 1.0);
        matrix.finalize();

        expected_solution = VectorObj<double>(4);
        expected_solution[0] = 2.0;
        expected_solution[1] = 3.0;
        expected_solution[2] = 2.0;
        expected_solution[3] = 1.0;

        b = VectorObj<double>(4);
        b[0] = 8.0;
        b[1] = 9.0;
        b[2] = 4.0;
        b[3] = 1.0;

        // Initial guess x0 = [0, 0, 0]
        x = VectorObj<double>(4, 0.0);
        solver = GMRES<double>();
    }

    SparseMatrixCSC<double> matrix;
    VectorObj<double> b;
    VectorObj<double> x;
    VectorObj<double> expected_solution;
    GMRES<double> solver;
};

TEST_F(GMRESTest, BasicSolverTest) {
    double tol = 1e-6;
    int maxIter = 100;
    int krylovDim = 3;
    
    solver.solve(matrix, b, x, maxIter, krylovDim, tol);

    for (int i = 0; i < expected_solution.size(); ++i) {
        EXPECT_NEAR(x[i], expected_solution[i], 1e-5) << "Mismatch at index " << i;
    }
}

TEST_F(GMRESTest, ZeroRHSTest) {
    VectorObj<double> zero_b(3, 0.0);
    VectorObj<double> zero_x(3, 0.0);
    
    solver.solve(matrix, zero_b, zero_x, 100, 3, 1e-6);

    // Solution should be zero vector
    EXPECT_NEAR(zero_x[0], 0.0, 1e-10);
    EXPECT_NEAR(zero_x[1], 0.0, 1e-10);
    EXPECT_NEAR(zero_x[2], 0.0, 1e-10);
    EXPECT_NEAR(zero_x[3], 0.0, 1e-10);
}

TEST_F(GMRESTest, ConvergenceTest) {
    VectorObj<double> x_initial = x;
    solver.solve(matrix, b, x, 100, 3, 1e-6);
    
    // Verify residual
    VectorObj<double> residual = b - matrix * x;
    double residual_norm = residual.L2norm();
    EXPECT_LT(residual_norm, 1e-6);
}

TEST_F(GMRESTest, DimensionMismatchTest) {
    VectorObj<double> wrong_size_x(3, 0.0);
    EXPECT_THROW(solver.solve(matrix, b, wrong_size_x, 100, 3, 1e-6), std::invalid_argument);
}

// TEST_F(GMRESTest, KrylovDimensionTest) {
//     // Test with different Krylov subspace dimensions
//     std::vector<int> krylov_dims = {1, 2, 3};
//     for (int dim : krylov_dims) {
//         VectorObj<double> x_test = x;
//         solver.solve(matrix, b, x_test, 100, dim, 1e-6);
        
//         VectorObj<double> residual = b - matrix * x_test;
//         EXPECT_LT(residual.L2norm(), 1e-6);
//     }
// }

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
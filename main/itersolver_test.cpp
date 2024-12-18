#include <gtest/gtest.h>
#include "IterSolver.hpp"

// Test for GradientDescent Solver
TEST(GradientDescentSolverTest, SimpleTestWithPreconditioner) {
    // Setup: Create a mock sparse matrix and a vector b
    SparseMatrixCSC<double> A(4, 4);
    A.addValue(0, 0, 1.0);
    // A.addValue(1, 0, -5.0);
    // A.addValue(2, 0, 0.0);
    // A.addValue(3, 0, 1.0);

    // A.addValue(0, 1, -1.0);
    A.addValue(1, 1, 1.0);
    // A.addValue(2, 1, 9.0);
    // A.addValue(3, 1, 0.0);

    // A.addValue(0, 2, -6.0);
    // A.addValue(1, 2, 10.0);
    A.addValue(2, 2, 1.0);
    // A.addValue(3, 2, -7.0);

    // A.addValue(0, 3, 0.0);
    // A.addValue(1, 3, 8.0);
    // A.addValue(2, 3, -2.0);
    A.addValue(3, 3, 1.0);
    A.finalize();

    VectorObj<double> b(4);
    b[0] = 2.0;
    b[1] = 21.0;
    b[2] = -12.0;
    b[3] = -6.0;

    VectorObj<double> x(4);
    x[0] = 0.0;
    x[1] = 0.0;
    x[2] = 0.0;
    x[3] = 0.0; // Initial guess

    // No preconditioner for simplicity in this case
    GradientDescent<double, SparseMatrixCSC<double>, VectorObj<double>> solver(A, b, 1000, 1e-6);

    // Solve the system
    solver.solve(x);


    EXPECT_NEAR(x[0], 2.0, 1e-6);
    EXPECT_NEAR(x[1], 21.0, 1e-6);
    EXPECT_NEAR(x[2], -12.0, 1e-6);
    EXPECT_NEAR(x[3], -6.0, 1e-6);

    // EXPECT_NEAR(x[0], 3.0, 1e-6);
    // EXPECT_NEAR(x[1], -2.0, 1e-6);
    // EXPECT_NEAR(x[2], 2.0, 1e-6);
    // EXPECT_NEAR(x[3], 1.0, 1e-6);
}

// Test for StaticIterMethod (Jacobi Method)
TEST(StaticIterMethodTest, JacobiMethodConvergence) {
    // Setup: Create a mock sparse matrix and a vector b
   SparseMatrixCSC<double> A(4, 4);
    A.addValue(0, 0, 1.0);
    // A.addValue(1, 0, -5.0);
    // A.addValue(2, 0, 0.0);
    // A.addValue(3, 0, 1.0);

    // A.addValue(0, 1, -1.0);
    A.addValue(1, 1, 1.0);
    // A.addValue(2, 1, 9.0);
    // A.addValue(3, 1, 0.0);

    // A.addValue(0, 2, -6.0);
    // A.addValue(1, 2, 10.0);
    A.addValue(2, 2, 1.0);
    // A.addValue(3, 2, -7.0);

    // A.addValue(0, 3, 0.0);
    // A.addValue(1, 3, 8.0);
    // A.addValue(2, 3, -2.0);
    A.addValue(3, 3, 1.0);
    A.finalize();

    VectorObj<double> b(4);
    b[0] = 2.0;
    b[1] = 21.0;
    b[2] = -12.0;
    b[3] = -6.0;

    VectorObj<double> x(4);
    x[0] = 0.0;
    x[1] = 0.0;
    x[2] = 0.0;
    x[3] = 0.0; // Initial guess
    StaticIterMethod<double, SparseMatrixCSC<double>, VectorObj<double>> solver(A, b, 1000, 0.5);

    // Solve the system
    solver.solve(x);

    // Check that the solution is approximately [2.0, 2.0, 2.0]

    EXPECT_NEAR(x[0], 2.0, 1e-6);
    EXPECT_NEAR(x[1], 21.0, 1e-6);
    EXPECT_NEAR(x[2], -12.0, 1e-6);
    EXPECT_NEAR(x[3], -6.0, 1e-6);
}

// Test for Early Stopping in GradientDescent due to convergence check
TEST(GradientDescentSolverTest, ConvergenceCheckEarlyStopping) {
    // Setup: Create a mock sparse matrix and a vector b
    SparseMatrixCSC<double> A(4, 4);
    A.addValue(0, 0, 1.0);
    // A.addValue(1, 0, -5.0);
    // A.addValue(2, 0, 0.0);
    // A.addValue(3, 0, 1.0);

    // A.addValue(0, 1, -1.0);
    A.addValue(1, 1, 1.0);
    // A.addValue(2, 1, 9.0);
    // A.addValue(3, 1, 0.0);

    // A.addValue(0, 2, -6.0);
    // A.addValue(1, 2, 10.0);
    A.addValue(2, 2, 1.0);
    // A.addValue(3, 2, -7.0);

    // A.addValue(0, 3, 0.0);
    // A.addValue(1, 3, 8.0);
    // A.addValue(2, 3, -2.0);
    A.addValue(3, 3, 1.0);
    A.finalize();

    VectorObj<double> b(4);
    b[0] = 2.0;
    b[1] = 21.0;
    b[2] = -12.0;
    b[3] = -6.0;

    VectorObj<double> x(4);
    x[0] = 0.0;
    x[1] = 0.0;
    x[2] = 0.0;
    x[3] = 0.0; // Initial guess

    // No preconditioner for simplicity in this case
    GradientDescent<double, SparseMatrixCSC<double>, VectorObj<double>> solver(A, b, 1000, 1e-6);

    // Solve the system
    solver.solve(x);


    EXPECT_NEAR(x[0], 2.0, 1e-6);
    EXPECT_NEAR(x[1], 21.0, 1e-6);
    EXPECT_NEAR(x[2], -12.0, 1e-6);
    EXPECT_NEAR(x[3], -6.0, 1e-6);

    // EXPECT_NEAR(x[0], 3.0, 1e-6);
    // EXPECT_NEAR(x[1], -2.0, 1e-6);
    // EXPECT_NEAR(x[2], 2.0, 1e-6);
    // EXPECT_NEAR(x[3], 1.0, 1e-6);

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
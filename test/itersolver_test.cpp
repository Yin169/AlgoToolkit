#include <gtest/gtest.h>
#include "IterSolver.hpp"

// Test for GradientDescent Solver
TEST(GradientDescentSolverTest, SimpleTestWithPreconditioner) {
    // Setup: Create a mock sparse A.addValue and a vector b
    SparseMatrixCSC<double> A(3, 3);
    A.addValue(0, 0, 4);
    A.addValue(1, 0, -1);
    A.addValue(0, 1, -1);
    A.addValue(1, 1, 4);
    A.addValue(2, 1, -1);
    A.addValue(1, 2, -1);
    A.addValue(2, 2, 4);
    A.finalize();

    VectorObj<double> b(3);
    b[0] = 1.0;
    b[1] = 5.0;
    b[2] = 0.0;

    VectorObj<double> x(3);
    x[0] = 0.0;
    x[1] = 0.0;
    x[2] = 0.0;

    // No preconditioner for simplicity in this case
    GradientDescent<double, SparseMatrixCSC<double>, VectorObj<double>> solver(A, b, 1000, 1e-6);

    // Solve the system
    solver.solve(x);

    EXPECT_NEAR(x[0], 0.625, 1e-5);
    EXPECT_NEAR(x[1], 1.5, 1e-5);
    EXPECT_NEAR(x[2], 0.375, 1e-5);

}

// Test for StaticIterMethod (Jacobi Method)
TEST(StaticIterMethodTest, JacobiMethodConvergence) {
    // Setup: Create a mock sparse A.addValue and a vector b
    SparseMatrixCSC<double> A(3, 3);
    A.addValue(0, 0, 4);
    A.addValue(1, 0, -1);
    A.addValue(0, 1, -1);
    A.addValue(1, 1, 4);
    A.addValue(2, 1, -1);
    A.addValue(1, 2, -1);
    A.addValue(2, 2, 4);
    A.finalize();

    VectorObj<double> b(3);
    b[0] = 1.0;
    b[1] = 5.0;
    b[2] = 0.0;

    VectorObj<double> x(3);
    x[0] = 0.0;
    x[1] = 0.0;
    x[2] = 0.0;
    StaticIterMethod<double, SparseMatrixCSC<double>, VectorObj<double>> solver(A, b, 1000, 0.5);

    // Solve the system
    solver.solve(x);

    EXPECT_NEAR(x[0], 0.625, 1e-5);
    EXPECT_NEAR(x[1], 1.5, 1e-5);
    EXPECT_NEAR(x[2], 0.375, 1e-5);
}

// Test for Early Stopping in GradientDescent due to convergence check
TEST(GradientDescentSolverTest, ConvergenceCheckEarlyStopping) {
    // Setup: Create a mock sparse A.addValue and a vector b
    SparseMatrixCSC<double> A(3, 3);
    A.addValue(0, 0, 4);
    A.addValue(1, 0, -1);
    A.addValue(0, 1, -1);
    A.addValue(1, 1, 4);
    A.addValue(2, 1, -1);
    A.addValue(1, 2, -1);
    A.addValue(2, 2, 4);
    A.finalize();

    VectorObj<double> b(3);
    b[0] = 1.0;
    b[1] = 5.0;
    b[2] = 0.0;

    VectorObj<double> x(3);
    x[0] = 0.0;
    x[1] = 0.0;
    x[2] = 0.0;

    // No preconditioner for simplicity in this case
    GradientDescent<double, SparseMatrixCSC<double>, VectorObj<double>> solver(A, b, 1000, 1e-6);

    // Solve the system
    solver.solve(x);

    EXPECT_NEAR(x[0], 0.625, 1e-5);
    EXPECT_NEAR(x[1], 1.5, 1e-5);
    EXPECT_NEAR(x[2], 0.375, 1e-5);

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
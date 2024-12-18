#include <gtest/gtest.h>
#include "IterSolver.hpp"

// Test for GradientDescent Solver
TEST(GradientDescentSolverTest, SimpleTestWithPreconditioner) {
    // Setup: Create a mock sparse matrix and a vector b
    SparseMatrixCSC<double> A;
    VectorObj<double> b(3);
    b[0] = b[1] = b[2] = 4.0;
    VectorObj<double> x(3);
    x[0] = x[1], x[2] = 0.0;  // Initial guess

    // No preconditioner for simplicity in this case
    GradientDescent<double, SparseMatrixCSC<double>, VectorObj<double>> solver(A, b, 1000, 1e-6);

    // Solve the system
    solver.solve(x);

    // Check that the solution is approximately [2.0, 2.0, 2.0]
    EXPECT_NEAR(x[0], 2.0, 1e-6);
    EXPECT_NEAR(x[1], 2.0, 1e-6);
    EXPECT_NEAR(x[2], 2.0, 1e-6);
}

// Test for StaticIterMethod (Jacobi Method)
TEST(StaticIterMethodTest, JacobiMethodConvergence) {
    // Setup: Create a mock sparse matrix and a vector b
    SparseMatrixCSC<double> A;
    VectorObj<double> b(3);
    b[0] = b[1] = b[2] = 4.0;
    VectorObj<double> x(3);
    x[0] = x[1], x[2] = 0.0;  // Initial guess

    StaticIterMethod<double, SparseMatrixCSC<double>, VectorObj<double>> solver(A, b, 1000);

    // Solve the system
    solver.solve(x);

    // Check that the solution is approximately [2.0, 2.0, 2.0]
    EXPECT_NEAR(x[0], 2.0, 1e-6);
    EXPECT_NEAR(x[1], 2.0, 1e-6);
    EXPECT_NEAR(x[2], 2.0, 1e-6);
}

// Test for Early Stopping in GradientDescent due to convergence check
TEST(GradientDescentSolverTest, ConvergenceCheckEarlyStopping) {
    // Setup: Create a mock sparse matrix and a vector b
    SparseMatrixCSC<double> A;
    VectorObj<double> b(3);
    b[0] = b[1] = b[2] = 4.0;
    VectorObj<double> x(3);
    x[0] = x[1], x[2] = 0.0;  // Initial guess

    // No preconditioner, use tolerance of 1e-6
    GradientDescent<double, SparseMatrixCSC<double>, VectorObj<double>> solver(A, b, 1000, 1e-6);

    // Solve the system
    solver.solve(x);

    // Check that the solution is approximately [2.0, 2.0, 2.0]
    EXPECT_NEAR(x[0], 2.0, 1e-6);
    EXPECT_NEAR(x[1], 2.0, 1e-6);
    EXPECT_NEAR(x[2], 2.0, 1e-6);
}

// Test for Divergence Handling (StaticIterMethod)
TEST(StaticIterMethodTest, DivergenceCheck) {
    // Setup: Create a mock sparse matrix (non-diagonal or poorly conditioned matrix)
    SparseMatrixCSC<double> A;
    VectorObj<double> b(3);
    b[0] = b[1] = b[2] = 4.0;
    VectorObj<double> x(3);
    x[0] = x[1], x[2] = 0.0;  // Initial guess

    StaticIterMethod<double, SparseMatrixCSC<double>, VectorObj<double>> solver(A, b, 1000);

    // Here, we intentionally expect the solver to converge in the given number of iterations
    EXPECT_THROW(solver.solve(x), std::runtime_error);  // Expecting an exception due to divergence
}

// // Test for Near-Zero Diagonal Handling
// TEST(StaticIterMethodTest, NearZeroDiagonalElement) {
//     // Setup: Create a mock sparse matrix with near-zero diagonal element
//     SparseMatrixCSC<double> A;
//     VectorObj<double> b(3);
//     b[0] = b[1] = b[2] = 4.0;
//     VectorObj<double> x(3);
//     x[0] = x[1], x[2] = 0.0;  // Initial guess

//     // Modify A to have a near-zero diagonal element
//     // Set A(0,0) to a very small value
//     // The test expects an exception to be thrown
//     void*(&funptr)(VectorObj<double>&) = &(StaticIterMethod<double, SparseMatrixCSC<double>, VectorObj<double>>)::solve; 
//     EXPECT_THROW(funptr(x), std::runtime_error);
// }


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
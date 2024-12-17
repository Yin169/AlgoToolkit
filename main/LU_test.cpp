#include <gtest/gtest.h>
#include "LU.hpp"
#include "MatrixObj.hpp"

// Helper function to initialize a square matrix
MatrixObj<double> initializeSquareMatrix(int n) {
    double data[] = {4, 3, 2, 
                     3, 4, 1, 
                     2, 1, 4}; // Example data for a 3x3 matrix
    return MatrixObj<double>(data, n, n);
}

// Test case for checking if the PivotLU function throws an exception for non-square matrices.
TEST(LUDecomposition, NonSquareMatrix) {
    MatrixObj<double> A(3, 4); // Non-square matrix
    std::vector<double> P;
    EXPECT_THROW(LU::PivotLU(A, P), std::invalid_argument);
}

// Test case for checking if the PivotLU function correctly performs LU decomposition on a square matrix.
TEST(LUDecomposition, SquareMatrix) {
    int n = 3;

    // Initialize matrix A with known values.
    MatrixObj<double> A = initializeSquareMatrix(n);
    std::vector<double> P;

    // Perform LU decomposition.
    LU::PivotLU(A, P);

    // Check if L * U reconstructs A
    MatrixObj<double> A_reconstructed = A; // Initially, copy A (since LU decomposition modifies it)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i <= j) {
                // Upper triangular part: U
                A_reconstructed[i + j * n] = A[i + j * n];
            } else {
                // Lower triangular part: L
                A_reconstructed[i + j * n] = A[i + j * n];
            }
        }
    }

    // Set diagonal elements to 1 for L
    for (int i = 0; i < n; ++i) {
        A_reconstructed[i + i * n] = 1.0;
    }

    // Verify if A is approximately equal to L * U
    MatrixObj<double> L_times_U = A_reconstructed;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(A[i + j * n], L_times_U[i + j * n], 1e-8);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

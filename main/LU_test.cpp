#include <gtest/gtest.h>
#include "LU.hpp"
#include "MatrixObj.hpp"

// Helper function to initialize a square matrix
MatrixObj<double> initializeSquareMatrix(int n) {
    std::vector<double> data({4, 3, 2, 3, 4, 1, 2, 1, 4}); // Example data for a 3x3 matrix
    return MatrixObj<double>(data, n, n);
}

// Test case for checking if the PivotLU function throws an exception for non-square matrices.
TEST(LUDecomposition, NonSquareMatrix) {
    MatrixObj<double> A(3, 4); // Non-square matrix
    std::vector<int> P;
    EXPECT_THROW(LU::PivotLU(A, P), std::invalid_argument);
}

// Test case for checking if the PivotLU function correctly performs LU decomposition on a square matrix.
TEST(LUDecomposition, SquareMatrix) {
    int n = 3;

    // Initialize matrix A with known values.
    MatrixObj<double> A = initializeSquareMatrix(n);
    MatrixObj<double> backup_A = A;
    std::vector<int> P;

    // Perform LU decomposition.
    LU::PivotLU(A, P);

    // Initialize L and U matrices
    MatrixObj<double> L(n, n);
    MatrixObj<double> U(n, n);

    // Reconstruct L and U from A
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i <= j) {
                // Upper triangular part: U
                U(i, j) = A(i, j);
            } else {
                // Lower triangular part: L
                L(i, j) = A(i, j) / A(i, i); // Assuming A(i, i) is non-zero
            }
        }
    }

    // Set diagonal elements of L to 1
    for (int i = 0; i < n; ++i) {
        L(i, i) = 1.0;
    }

    // Verify if A is approximately equal to L * U
    MatrixObj<double> L_times_U = L * U;

    for (int i = 0 ; i < P.size(); i++){
        L_times_U.swapRows(i, P[i]);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(backup_A(i, j), L_times_U(i, j), 1e-8);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#include <gtest/gtest.h>
#include "ILU.hpp"
#include <cmath>

class ILUTest : public ::testing::Test {
protected:
    using MatrixType = SparseMatrixCSC<double>;
    using VectorType = VectorObj<double>;
    
    // Helper function to create a simple test matrix
    MatrixType createTestMatrix() {
        MatrixType A(3, 3);
        A.addValue(0, 0, 4.0);  A.addValue(0, 1, -1.0); A.addValue(0, 2, 0.0);
        A.addValue(1, 0, -1.0); A.addValue(1, 1, 4.0);  A.addValue(1, 2, -1.0);
        A.addValue(2, 0, 0.0);  A.addValue(2, 1, -1.0); A.addValue(2, 2, 4.0);
        A.finalize();
        return A;
    }
    
    // Helper function to check if two matrices are approximately equal
    bool areMatricesEqual(const MatrixType& A, const MatrixType& B, double tol = 1e-10) {
        if (A.getRows() != B.getRows() || A.getCols() != B.getCols()) {
            return false;
        }
        
        for (int i = 0; i < A.getRows(); ++i) {
            for (int j = 0; j < A.getCols(); ++j) {
                if (std::abs(A(i,j) - B(i,j)) > tol) {
                    return false;
                }
            }
        }
        return true;
    }
    
    // Helper function to check if two vectors are approximately equal
    bool areVectorsEqual(const VectorType& a, const VectorType& b, double tol = 1e-10) {
        if (a.size() != b.size()) {
            return false;
        }
        
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(a[i] - b[i]) > tol) {
                return false;
            }
        }
        return true;
    }
};

// Test basic initialization
TEST_F(ILUTest, Initialization) {
    ILUPreconditioner<double> ilu;
    EXPECT_NO_THROW(ilu.compute(createTestMatrix()));
}

// Test for non-square matrix
TEST_F(ILUTest, NonSquareMatrix) {
    MatrixType nonsquare(3, 4);
    ILUPreconditioner<double> ilu;
    EXPECT_THROW(ilu.compute(nonsquare), std::invalid_argument);
}

// Test singular matrix detection
TEST_F(ILUTest, SingularMatrix) {
    MatrixType singular(3, 3);
    singular.addValue(0, 0, 0.0); singular.addValue(0, 1, 1.0);
    singular.addValue(1, 0, 0.0); singular.addValue(1, 1, 0.0);
    singular.addValue(2, 0, 0.0); singular.addValue(2, 1, 0.0);
    singular.finalize();
    
    ILUPreconditioner<double> ilu;
    EXPECT_THROW(ilu.compute(singular), std::runtime_error);
}

// Test solving a system
TEST_F(ILUTest, SolveSystem) {
    ILUPreconditioner<double> ilu;
    MatrixType A = createTestMatrix();
    ilu.compute(A);
    
    // Create a known solution
    VectorType x_true(3);
    x_true[0] = 1.0;
    x_true[1] = 2.0;
    x_true[2] = 3.0;
    
    // Compute b = Ax
    VectorType b(3);
    for (int i = 0; i < 3; ++i) {
        double sum = 0.0;
        for (int j = 0; j < 3; ++j) {
            sum += A(i,j) * x_true[j];
        }
        b[i] = sum;
    }
    
    // Solve the system
    VectorType x = ilu.solve(b);
    
    // Check if the solution is correct
    EXPECT_TRUE(areVectorsEqual(x, x_true));
}

// Test L and U factors properties
TEST_F(ILUTest, FactorsProperties) {
    ILUPreconditioner<double> ilu;
    ilu.compute(createTestMatrix());
    
    const auto& L = ilu.getLFactor();
    const auto& U = ilu.getUFactor();
    
    // Check dimensions
    EXPECT_EQ(L.getRows(), 3);
    EXPECT_EQ(L.getCols(), 3);
    EXPECT_EQ(U.getRows(), 3);
    EXPECT_EQ(U.getCols(), 3);
    
    // Check L's diagonal elements are 1
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(L(i,i), 1.0, 1e-10);
    }
    
    // Check L is lower triangular
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            EXPECT_NEAR(L(i,j), 0.0, 1e-10);
        }
    }
    
    // Check U is upper triangular
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < i; ++j) {
            EXPECT_NEAR(U(i,j), 0.0, 1e-10);
        }
    }
}

// Test solving without computing first
TEST_F(ILUTest, SolveWithoutCompute) {
    ILUPreconditioner<double> ilu;
    VectorType b(3, 1.0);
    EXPECT_THROW(ilu.solve(b), std::runtime_error);
}

// Test solving with wrong size vector
TEST_F(ILUTest, SolveWrongSize) {
    ILUPreconditioner<double> ilu;
    ilu.compute(createTestMatrix());
    VectorType b(4, 1.0);  // Wrong size
    EXPECT_THROW(ilu.solve(b), std::invalid_argument);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
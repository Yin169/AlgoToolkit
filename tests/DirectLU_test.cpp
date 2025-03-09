#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cmath>
#include <stdexcept>
#include "ILU.hpp"
#include "../src/Obj/VectorObj.hpp"
#include "../src/Obj/SparseObj.hpp"
#include "../src/SparseLinearAlgebra/DirectSolve.hpp"

namespace {

class LUSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple 3x3 sparse matrix for testing
        A = SparseMatrixCSC<double>(3, 3);
        A.addValue(0, 0, 4.0);
        A.addValue(0, 1, 3.0);
        A.addValue(1, 0, 6.0);
        A.addValue(1, 1, 3.0);
        A.addValue(1, 2, 1.0);
        A.addValue(2, 0, 2.0);
        A.addValue(2, 2, 3.0);
        A.finalize();
        
        ILUPreconditioner<double> ilu;
        ilu.compute(A);
        
        expectedL = ilu.getLFactor();
        expectedU = ilu.getUFactor();
    }

    // Helper function to compare sparse matrices
    bool compareSparseMatrices(const SparseMatrixCSC<double>& a, const SparseMatrixCSC<double>& b, double epsilon = 1e-10) {
        if (a.getRows() != b.getRows() || a.getCols() != b.getCols()) {
            return false;
        }
        
        for (int j = 0; j < a.getCols(); j++) {
            for (int i = 0; i < a.getRows(); i++) {
                if (std::abs(a(i, j) - b(i, j)) > epsilon) {
                    return false;
                }
            }
        }
        return true;
    }

    SparseMatrixCSC<double> A;
    SparseMatrixCSC<double> expectedL;
    SparseMatrixCSC<double> expectedU;
};

// Test successful LU decomposition
TEST_F(LUSolverTest, SuccessfulDecomposition) {
    // Create LU solver
    LUSolver<double> solver(A);
    
    // Get L and U matrices
    const SparseMatrixCSC<double>& L = solver.getL();
    const SparseMatrixCSC<double>& U = solver.getU();
    
    // Check dimensions
    EXPECT_EQ(L.getRows(), 3);
    EXPECT_EQ(L.getCols(), 3);
    EXPECT_EQ(U.getRows(), 3);
    EXPECT_EQ(U.getCols(), 3);
    
    // Check L and U matrices against expected values
    EXPECT_TRUE(compareSparseMatrices(L, expectedL));
    EXPECT_TRUE(compareSparseMatrices(U, expectedU));
    
    // Verify that L*U = A
    SparseMatrixCSC<double> product = L * U;
    EXPECT_TRUE(compareSparseMatrices(product, A));
}

// Test solving a linear system
TEST_F(LUSolverTest, SolveLinearSystem) {
    // Create LU solver
    LUSolver<double> solver(A);
    
    // Create right-hand side vector
    VectorObj<double> b(3);
    b[0] = 1.0;
    b[1] = 2.0;
    b[2] = 3.0;
    
    // Create a copy for verification
    VectorObj<double> b_copy = b;
    
    // Solve the system
    solver.solve(b);
    
    // Verify the solution by multiplying A*x = b_copy
    VectorObj<double> result = A * b;
    
    // Check if the result matches the original b
    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(result[i], b_copy[i], 1e-10);
    }
}

// Test non-square matrix
TEST_F(LUSolverTest, NonSquareMatrix) {
    // Create a non-square matrix
    SparseMatrixCSC<double> nonSquare(3, 2);
    nonSquare.addValue(0, 0, 1.0);
    nonSquare.addValue(1, 1, 2.0);
    nonSquare.finalize();
    
    // Expect exception when creating solver
    EXPECT_THROW(LUSolver<double> solver(nonSquare), std::invalid_argument);
}

// Test singular matrix
TEST_F(LUSolverTest, SingularMatrix) {
    // Create a singular matrix (with a zero on the diagonal)
    SparseMatrixCSC<double> singular(3, 3);
    singular.addValue(0, 0, 1.0);
    singular.addValue(1, 1, 0.0); // Zero pivot
    singular.addValue(2, 2, 3.0);
    singular.finalize();
    
    // Expect exception when creating solver
    EXPECT_THROW(LUSolver<double> solver(singular), std::runtime_error);
}

// Test with different data type (float)
TEST_F(LUSolverTest, FloatDataType) {
    // Create a matrix with float values
    SparseMatrixCSC<float> floatMatrix(2, 2);
    floatMatrix.addValue(0, 0, 2.0f);
    floatMatrix.addValue(0, 1, 1.0f);
    floatMatrix.addValue(1, 0, 1.0f);
    floatMatrix.addValue(1, 1, 3.0f);
    floatMatrix.finalize();
    
    // Create solver
    LUSolver<float> solver(floatMatrix);
    
    // Get L and U matrices
    const SparseMatrixCSC<float>& L = solver.getL();
    const SparseMatrixCSC<float>& U = solver.getU();
    
    // Check dimensions
    EXPECT_EQ(L.getRows(), 2);
    EXPECT_EQ(L.getCols(), 2);
    EXPECT_EQ(U.getRows(), 2);
    EXPECT_EQ(U.getCols(), 2);
    
    // Expected values for L
    EXPECT_FLOAT_EQ(L(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(L(1, 0), 0.5f);
    EXPECT_FLOAT_EQ(L(1, 1), 1.0f);
    
    // Expected values for U
    EXPECT_FLOAT_EQ(U(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(U(0, 1), 1.0f);
    EXPECT_FLOAT_EQ(U(1, 1), 2.5f);
}

// Test with a larger sparse matrix
TEST_F(LUSolverTest, LargerSparseMatrix) {
    // Create a larger sparse matrix (5x5)
    SparseMatrixCSC<double> largeMatrix(5, 5);
    largeMatrix.addValue(0, 0, 2.0);
    largeMatrix.addValue(0, 1, -1.0);
    largeMatrix.addValue(1, 0, -1.0);
    largeMatrix.addValue(1, 1, 2.0);
    largeMatrix.addValue(1, 2, -1.0);
    largeMatrix.addValue(2, 1, -1.0);
    largeMatrix.addValue(2, 2, 2.0);
    largeMatrix.addValue(2, 3, -1.0);
    largeMatrix.addValue(3, 2, -1.0);
    largeMatrix.addValue(3, 3, 2.0);
    largeMatrix.addValue(3, 4, -1.0);
    largeMatrix.addValue(4, 3, -1.0);
    largeMatrix.addValue(4, 4, 2.0);
    largeMatrix.finalize();
    
    // Create solver
    EXPECT_NO_THROW({
        LUSolver<double> solver(largeMatrix);
        
        // Verify dimensions
        const SparseMatrixCSC<double>& L = solver.getL();
        const SparseMatrixCSC<double>& U = solver.getU();
        EXPECT_EQ(L.getRows(), 5);
        EXPECT_EQ(L.getCols(), 5);
        EXPECT_EQ(U.getRows(), 5);
        EXPECT_EQ(U.getCols(), 5);
        
        // Verify L*U = A
        SparseMatrixCSC<double> product = L * U;
        EXPECT_TRUE(compareSparseMatrices(product, largeMatrix));
    });
}

} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
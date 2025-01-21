#include <gtest/gtest.h>
#include "ILU.hpp"
#include "SparseObj.hpp"
#include "DenseObj.hpp"
#include <cmath>

class ILUPreconditionerTest : public ::testing::Test {
protected:
    const double tolerance = 1e-8;
    
    bool isClose(double a, double b) {
        return std::abs(a - b) < tolerance;
    }

    bool vectorIsClose(const VectorObj<double>& a, const VectorObj<double>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); i++) {
            if (!isClose(a[i], b[i])) return false;
        }
        return true;
    }
};

TEST_F(ILUPreconditionerTest, SparseMatrixSimple2x2) {
    // Create a simple 2x2 sparse matrix
    SparseMatrixCSC<double> A(2, 2);
    A.addValue(0, 0, 2.0);
    A.addValue(0, 1, 1.0);
    A.addValue(1, 0, 1.0);
    A.addValue(1, 1, 3.0);
    A.finalize();

    ILUPreconditioner<double, SparseMatrixCSC<double>> ilu;
    ilu.compute(A);

    // Test with a known right-hand side
    VectorObj<double> b(2);
    b[0] = 3.0;
    b[1] = 4.0;

    VectorObj<double> x = ilu.solve(b);

    // Expected solution for the system Ax = b
    VectorObj<double> expected(2);
    expected[0] = 1.0;
    expected[1] = 1.0;

    EXPECT_TRUE(vectorIsClose(x, expected));
}

TEST_F(ILUPreconditionerTest, DenseMatrixSimple2x2) {
    // Create a simple 2x2 dense matrix
    DenseObj<double> A(2, 2);
    A(0, 0) = 2.0; A(0, 1) = 1.0;
    A(1, 0) = 1.0; A(1, 1) = 3.0;

    ILUPreconditioner<double, DenseObj<double>> ilu;
    ilu.compute(A);

    VectorObj<double> b(2);
    b[0] = 3.0;
    b[1] = 4.0;

    VectorObj<double> x = ilu.solve(b);

    VectorObj<double> expected(2);
    expected[0] = 1.0;
    expected[1] = 1.0;

    EXPECT_TRUE(vectorIsClose(x, expected));
}

TEST_F(ILUPreconditionerTest, SparseMatrixLarger) {
    // Create a 3x3 sparse matrix
    SparseMatrixCSC<double> A(3, 3);
    A.addValue(0, 0, 4.0);
    A.addValue(0, 1, -1.0);
    A.addValue(0, 2, 0.0);
    A.addValue(1, 0, -1.0);
    A.addValue(1, 1, 4.0);
    A.addValue(1, 2, -1.0);
    A.addValue(2, 0, 0.0);
    A.addValue(2, 1, -1.0);
    A.addValue(2, 2, 4.0);
    A.finalize();

    ILUPreconditioner<double, SparseMatrixCSC<double>> ilu;
    ilu.compute(A);

    VectorObj<double> b(3);
    b[0] = 1.0;
    b[1] = 5.0;
    b[2] = 0.0;

    VectorObj<double> x = ilu.solve(b);

    // Verify that Ax ≈ b
    VectorObj<double> Ax(3);
    for (int i = 0; i < 3; i++) {
        double sum = 0.0;
        for (int j = A.col_ptr[i]; j < A.col_ptr[i + 1]; j++) {
            sum += A.values[j] * x[A.row_indices[j]];
        }
        Ax[i] = sum;
    }

    EXPECT_TRUE(vectorIsClose(Ax, b));
}

TEST_F(ILUPreconditionerTest, NonSquareMatrixThrows) {
    SparseMatrixCSC<double> A(2, 3);
    ILUPreconditioner<double, SparseMatrixCSC<double>> ilu;
    
    EXPECT_THROW(ilu.compute(A), std::invalid_argument);

    DenseObj<double> B(2, 3);
    ILUPreconditioner<double, DenseObj<double>> ilu_dense;
    
    EXPECT_THROW(ilu_dense.compute(B), std::invalid_argument);
}

TEST_F(ILUPreconditionerTest, ZeroDiagonalHandling) {
    DenseObj<double> A(2, 2);
    A(0, 0) = 0.0; A(0, 1) = 1.0;
    A(1, 0) = 1.0; A(1, 1) = 1.0;

    ILUPreconditioner<double, DenseObj<double>> ilu;
    
    // This should either throw an exception or handle zero diagonal elements gracefully
    // depending on your implementation's requirements
    EXPECT_THROW(ilu.compute(A), std::runtime_error);
}

// TEST_F(ILUPreconditionerTest, SparsityPatternPreservation) {
//     // Create a sparse matrix with a specific sparsity pattern
//     SparseMatrixCSC<double> A(3, 3);
//     A.addValue(0, 0, 4.0);
//     A.addValue(1, 1, 4.0);
//     A.addValue(2, 2, 4.0);
//     A.addValue(1, 0, -1.0);
//     A.finalize();

//     ILUPreconditioner<double, SparseMatrixCSC<double>> ilu;
//     ilu.compute(A);

//     VectorObj<double> b(3);
//     b[0] = 1.0;
//     b[1] = 2.0;
//     b[2] = 3.0;

//     VectorObj<double> x = ilu.solve(b);

//     // Verify solution maintains sparsity pattern
//     VectorObj<double> Ax(3);
//     for (int i = 0; i < 3; i++) {
//         double sum = 0.0;
//         for (int j = A.col_ptr[i]; j < A.col_ptr[i + 1]; j++) {
//             sum += A.values[j] * x[A.row_indices[j]];
//         }
//         Ax[i] = sum;
//     }

//     EXPECT_TRUE(vectorIsClose(Ax, b));
// }

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
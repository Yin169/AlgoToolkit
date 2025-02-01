#include <gtest/gtest.h>
#include "../../../src/LinearAlgebra/Factorized/basic.hpp"
#include "../../../src/Obj/DenseObj.hpp"
#include "../../../src/Obj/VectorObj.hpp"
#include <cmath>

class SVDTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup code if needed
    }

    // Helper function to check if two matrices are approximately equal
    template<typename TNum>
    bool isApproxEqual(const DenseObj<TNum>& A, const DenseObj<TNum>& B, TNum tolerance = 1e-10) {
        if (A.getRows() != B.getRows() || A.getCols() != B.getCols()) return false;
        
        for (int i = 0; i < A.getRows(); ++i) {
            for (int j = 0; j < A.getCols(); ++j) {
                if (std::abs(A(i,j) - B(i,j)) > tolerance) return false;
            }
        }
        return true;
    }

    // Helper function to verify SVD properties
    template<typename TNum>
    void verifySVD(const DenseObj<TNum>& A, const DenseObj<TNum>& U, 
                   const DenseObj<TNum>& S, const DenseObj<TNum>& V) {
        // Check dimensions
        ASSERT_EQ(U.getRows(), A.getRows());
        ASSERT_EQ(U.getCols(), A.getRows());
        ASSERT_EQ(S.getRows(), A.getRows());
        ASSERT_EQ(S.getCols(), A.getCols());
        ASSERT_EQ(V.getRows(), A.getCols());
        ASSERT_EQ(V.getCols(), A.getCols());

        // Check orthogonality of U and V
        DenseObj<TNum> UtU = U.Transpose() * U;
        DenseObj<TNum> VtV = V.Transpose() * V;
        DenseObj<TNum> I_m = basic::genUnitMat<TNum, DenseObj<TNum>>(U.getRows());
        DenseObj<TNum> I_n = basic::genUnitMat<TNum, DenseObj<TNum>>(V.getRows());

        EXPECT_TRUE(isApproxEqual(UtU, I_m));
        EXPECT_TRUE(isApproxEqual(VtV, I_n));

        // Check reconstruction A = U*S*V^T
        DenseObj<TNum> reconstructed = U * S * V.Transpose();
        EXPECT_TRUE(isApproxEqual(reconstructed, A));

        // Check that singular values are non-negative and in descending order
        for (int i = 0; i < std::min(S.getRows(), S.getCols()) - 1; ++i) {
            EXPECT_GE(S(i,i), 0);
            EXPECT_GE(S(i,i), S(i+1,i+1));
        }
    }
};

TEST_F(SVDTest, Identity2x2Matrix) {
    DenseObj<double> A(2, 2);
    A(0,0) = 1.0; A(0,1) = 0.0;
    A(1,0) = 0.0; A(1,1) = 1.0;

    DenseObj<double> U, S, V;
    basic::SVD(A, U, S, V);

    verifySVD(A, U, S, V);

    // For identity matrix, singular values should all be 1
    EXPECT_NEAR(S(0,0), 1.0, 1e-10);
    EXPECT_NEAR(S(1,1), 1.0, 1e-10);
}

TEST_F(SVDTest, Simple2x2Matrix) {
    DenseObj<double> A(2, 2);
    A(0,0) = 4.0; A(0,1) = 0.0;
    A(1,0) = 3.0; A(1,1) = -5.0;

    DenseObj<double> U, S, V;
    basic::SVD(A, U, S, V);

    verifySVD(A, U, S, V);
}

TEST_F(SVDTest, RectangularMatrix) {
    DenseObj<double> A(3, 2);
    A(0,0) = 1.0; A(0,1) = 2.0;
    A(1,0) = 3.0; A(1,1) = 4.0;
    A(2,0) = 5.0; A(2,1) = 6.0;

    DenseObj<double> U, S, V;
    basic::SVD(A, U, S, V);

    verifySVD(A, U, S, V);
}

TEST_F(SVDTest, ZeroMatrix) {
    DenseObj<double> A(2, 2);
    // Zero matrix - all elements are 0

    DenseObj<double> U, S, V;
    basic::SVD(A, U, S, V);

    verifySVD(A, U, S, V);

    // All singular values should be 0
    EXPECT_NEAR(S(0,0), 0.0, 1e-10);
    EXPECT_NEAR(S(1,1), 0.0, 1e-10);
}

TEST_F(SVDTest, InvalidDimensions) {
    DenseObj<double> A(0, 0);
    DenseObj<double> U, S, V;

    EXPECT_THROW(basic::SVD(A, U, S, V), std::invalid_argument);
}
#include <cmath>
#include <iostream>
#include <vector>
#include "basic.hpp"
#include "gtest/gtest.h"

namespace {

template <typename MatrixType, typename VectorType>
class BasicTest : public ::testing::Test {
protected:
    MatrixType A;
    VectorType b;
    double setA[3 * 3] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double setB[3] = {1, 0, 0};

    void SetUp() override {
        A = MatrixType(3, 3);
        std::copy(std::begin(setA), std::end(setA), A.data());
        b = VectorType(setB, 3);
    }
};

using MyTest = BasicTest<DenseObj<double>, VectorObj<double>>;

TEST_F(MyTest, RayleighQuotientTest) {
    double quotient = basic::rayleighQuotient<double, DenseObj<double>>(A, b);
    ASSERT_NEAR(quotient, 1, 1e-6) << "Rayleigh quotient test failed. Actual: " << quotient;
}

TEST_F(MyTest, PowerIterTest) {
    VectorObj<double> b_init = b;
    basic::powerIter<double, DenseObj<double>>(A, b, 10);
    for (int i = 0; i < b.size(); ++i) {
        ASSERT_NEAR(b_init[i], b[i], 1e-6) << "Power iteration test failed at index " << i << ". Actual: " << b[i];
    }
}

TEST_F(MyTest, GramSchmidtTest) {
    int m = A.getCols();
    std::vector<VectorObj<double>> orthSet(m);
    basic::gramSchmidt<double, DenseObj<double>>(A, orthSet);

    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < m; ++j) {
            auto dot_product = orthSet[i] * orthSet[j];
            ASSERT_NEAR(dot_product, 0.0, 1e-6) << "Gram-Schmidt test failed at (" << i << ", " << j << "). Actual: " << dot_product;
        }
    }
}

TEST_F(MyTest, GenUnitVecTest) {
    VectorObj<double> e = basic::genUnitVec<double>(1, 3);
    ASSERT_NEAR(e[0], 0, 1e-6);
    ASSERT_NEAR(e[1], 1, 1e-6);
    ASSERT_NEAR(e[2], 0, 1e-6);
}

TEST_F(MyTest, GenUnitMatTest) {
    DenseObj<double> I = basic::genUnitMat<double, DenseObj<double>>(3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ASSERT_NEAR(I[i * 3 + j], (i == j) ? 1 : 0, 1e-6) << "Matrix identity check failed at (" << i << ", " << j << ")";
        }
    }
}

TEST_F(MyTest, SignTest) {
    ASSERT_GT(basic::sign(5.0), 0);
    ASSERT_LT(basic::sign(-3.0), 0);
}

TEST_F(MyTest, HouseholderTransformTest) {
    DenseObj<double> H;
    basic::householderTransform<double, DenseObj<double>>(A, H, 0);
    DenseObj<double> Ht = H.Transpose();
    DenseObj<double> Res = H * Ht;

    for (int i = 0; i < A.getRows(); ++i) {
        for (int j = 0; j < A.getCols(); ++j) {
            bool is_diag = (i == j);
            double expected = is_diag ? 1.0 : 0.0;
            ASSERT_NEAR(Res[i * H.getRows() + j], expected, 1e-6) << "Householder test failed at (" << i << ", " << j << ")";
            ASSERT_NEAR(H[i * H.getRows() + j], Ht[i * Ht.getRows() + j], 1e-6);
        }
    }
}

TEST_F(MyTest, QrFactorizationTest) {
    DenseObj<double> Q, R;
    basic::qrFactorization<double, DenseObj<double>>(A, Q, R);

    // Check if R is upper triangular
    for (int i = 1; i < A.getCols(); ++i) {
        for (int j = 0; j < i; ++j) {
            ASSERT_NEAR(R[i * A.getRows() + j], 0.0, 1e-6) << "QR factorization test failed: R is not upper triangular at (" << i << ", " << j << ")";
        }
    }

    // Check if A = QR
    DenseObj<double> A_recons = Q * R;
    for (int i = 0; i < A.getCols(); ++i) {
        for (int j = 0; j < A.getRows(); ++j) {
            ASSERT_NEAR(A_recons[i * A_recons.getRows() + j], A[i * A.getRows() + j], 1e-6) 
                << "QR factorization test failed: A != QR at (" << i << ", " << j << ")";
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

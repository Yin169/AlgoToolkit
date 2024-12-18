#include <cmath>
#include <iostream>
#include "DenseObj.hpp"
#include "basic.hpp"
#include "gtest/gtest.h"

namespace {

class BasicTest : public ::testing::Test {
protected:
    DenseObj<double> A;
    VectorObj<double> b;
    double setA[3*3] = {1,0,0, 0,1,0, 0,0,1};
    double setB[3] = {1, 0, 0};

    void SetUp() override {
        A = DenseObj<double>(3, 3);
        std::copy(std::begin(setA), std::end(setA), A.data());
        b = VectorObj<double>(setB, 3);
    }
};

TEST_F(BasicTest, RayleighQuotientTest) {
    double quotient = basic::rayleighQuotient<double>(A, b);
    ASSERT_NEAR(quotient, 1, 1e-6) << "Rayleigh quotient test failed. Actual: " << quotient;
}

TEST_F(BasicTest, PowerIterTest) {
    VectorObj<double> b_init = b;
    basic::powerIter(A, b, 10);
    for (int i = 0; i < b.size(); ++i) { 
        ASSERT_NEAR(b_init[i], b[i], 1e-6) << "Power iteration test failed at index " << i << ". Actual: " << b[i];
    }
}

TEST_F(BasicTest, GramSmithTest) {
    std::cout << "Starting Gram-Schmidt test..." << std::endl;
    int m = A.getCols();
    std::vector<VectorObj<double>> orthSet(m);
    basic::gramSchmidt(A, orthSet);

    for (int i = 0; i < m; ++i) {
        std::cout << "Checking orthogonality for vector " << i << "..." << std::endl;
        for (int j = i + 1; j < m; ++j) {
            auto dot_product = orthSet[i] * orthSet[j];
            ASSERT_NEAR(dot_product, 0.0, 1e-6) << "Gram-Schmidt test failed at (" << i << ", " << j << "). Actual: " << dot_product;
        }
    }
    std::cout << "Gram-Schmidt test completed successfully." << std::endl;
}

TEST_F(BasicTest, GenUnitvecTest) {
    VectorObj<double> e;
    e = basic::genUnitVec<double>(1, 3);
    ASSERT_NEAR(e[0], 0, 1e-6);
    ASSERT_NEAR(e[1], 1, 1e-6);
    ASSERT_NEAR(e[2], 0, 1e-6);
}

TEST_F(BasicTest, GenUnitMatxTest) {
    DenseObj<double> I;
    I = basic::genUnitMat<double>(3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ASSERT_NEAR(I[i * 3 + j], (i == j) ? 1 : 0, 1e-6) << "Matrix identity check failed at (" << i << ", " << j << ")";
        }
    }
}

TEST_F(BasicTest, SignTest) {
    ASSERT_GT(basic::sign(5.0), 0);
    ASSERT_LT(basic::sign(-3.0), 0);
}

TEST_F(BasicTest, HouseHTest) {
    DenseObj<double> H;
    basic::householderTransform(A, H, 0);
    DenseObj<double> Ht = H.Transpose();
    DenseObj<double> Res = H * Ht;

    for (int i = 0; i < A.getRows(); ++i) {
        for (int j = 0; j < A.getCols(); ++j) {
            bool is_diag = (i == j);
            double expected = is_diag ? 1.0 : 0.0;
            ASSERT_NEAR(Res[i * H.getRows() + j], expected, 1e-6) << "HouseH test failed at (" << i << ", " << j << ")";
            ASSERT_NEAR(H[i * H.getRows() + j], Ht[i * Ht.getRows() + j], 1e-6);
        }
    }
}

TEST_F(BasicTest, qrFactorizationTest) {
    DenseObj<double> Q, R;
    basic::qrFactorization(A, Q, R);

    // Check if R is upper triangular
    for (int i = 1; i < A.getCols(); ++i) {
        for (int j = 0; j < i; ++j) {
            ASSERT_NEAR(R[i * A.getRows() + j], 0.0, 1e-6) << "qrFactorizationTest failed: R is not upper triangular at (" << i << ", " << j << ")";
        }
    }

    // Check if A = QR
    DenseObj<double> A_recons = Q * R;
    for (int i = 0; i < A.getCols(); ++i) {
        for (int j = 0; j < A.getRows(); ++j) {
            ASSERT_NEAR(A_recons[i * A_recons.getRows() + j], A[i * A.getRows() + j], 1e-6) 
                << "qrFactorizationTest failed: A != QR at (" << i << ", " << j << ")";
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

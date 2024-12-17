#include "../src/LinearAlgebra/Solver/IterSolver.hpp"
#include "gtest/gtest.h"

// Helper function to create an identity matrix of a given size.
template<typename TNum>
MatrixObj<TNum> CreateIdentityMatrix(int size) {
    MatrixObj<TNum> identity(size, size);
    for (int i = 0; i < size; ++i) {
        identity(i , i) = 1; // Set the diagonal elements to 1.
    }
    return identity;
}

// Helper function to create a vector with a specified value.
template<typename TNum>
VectorObj<TNum> CreateVector(int size, TNum value) {
    VectorObj<TNum> vec(size);
    std::fill(vec.begin(), vec.end(), value);
    return vec;
}

// Test Fixture for GradientDesent
class GradientDesentTest : public ::testing::Test {
protected:
    int size_ = 100;
    MatrixObj<double> P = CreateIdentityMatrix<double>(size_);
    MatrixObj<double> A = CreateIdentityMatrix<double>(size_);
    VectorObj<double> b = CreateVector<double>(size_, 1.0);
    VectorObj<double> x = CreateVector<double>(size_, 0.0);
    std::unique_ptr<GradientDesent<double>> gradientDesent;

    void SetUp() override {
        gradientDesent = std::make_unique<GradientDesent<double>>(P, A, b, 100);
    }
};

// Test the constructor of GradientDesent.
TEST_F(GradientDesentTest, Constructor) {
    EXPECT_EQ(gradientDesent->getIter(), 100);
}

// Test the calculation of gradient.
TEST_F(GradientDesentTest, CalGrad) {
    VectorObj<double> expectedGradient = b - (A * x);
    VectorObj<double> actualGradient = gradientDesent->calGrad(x);
    EXPECT_NEAR((actualGradient - expectedGradient).L2norm(), 0.0, 1e-8);
}

// Test the update step in GradientDesent.
TEST_F(GradientDesentTest, Update) {
    VectorObj<double> expectedX = CreateVector<double>(size_, 1.0);
    x = CreateVector<double>(size_, 0.0);
    gradientDesent->callUpdate(x);
    EXPECT_NEAR((x - expectedX).L2norm(), 0.0, 1e-8);
}

// Test Fixture for StaticIterMethod
class StaticIterMethodTest : public ::testing::Test {
protected:
    int size_ = 100;
    MatrixObj<double> A = CreateIdentityMatrix<double>(size_);
    VectorObj<double> b = CreateVector<double>(size_, 1.0);
    VectorObj<double> x = CreateVector<double>(size_, 0.0);
    std::unique_ptr<StaticIterMethod<double>> staticIterMethod;

    void SetUp() override {
        staticIterMethod = std::make_unique<StaticIterMethod<double>>(A, A, b, 100);
    }
};

// Test the constructor of StaticIterMethod.
TEST_F(StaticIterMethodTest, Constructor) {
    EXPECT_EQ(staticIterMethod->getIter(), 100);
}

// Test the Substitution method of StaticIterMethod.
TEST_F(StaticIterMethodTest, Substitution) {
    MatrixObj<double> L = CreateIdentityMatrix<double>(size_);
    VectorObj<double> result = staticIterMethod->Substitution(b, L, true);
    EXPECT_NEAR((result - b).L2norm(), 0.0, 1e-8);  // L is identity, so result should be b
}

// Test the update step in StaticIterMethod.
TEST_F(StaticIterMethodTest, Update) {
    VectorObj<double> expectedX = CreateVector<double>(size_, 1.0);
    x = CreateVector<double>(size_, 0.0);
    staticIterMethod->callUpdate(x);
    EXPECT_NEAR((x - expectedX).L2norm(), 0.0, 1e-8);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

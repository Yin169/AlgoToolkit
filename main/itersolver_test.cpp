#include "../src/LinearAlgebra/Solver/IterSolver.hpp"
#include "gtest/gtest.h"

// Helper function to create an identity matrix of a given size.
template<typename TNum>
DenseObj<TNum> CreateIdentityMatrix(int size) {
    DenseObj<TNum> identity(size, size);
    for (int i = 0; i < size; ++i) {
        identity(i , i) = 1; // Set the diagonal elements to 1.
    }
    return identity;
}

// Helper function to create a vector with a specified value.
template<typename TNum>
VectorObj<TNum> CreateVector(int size, TNum value) {
    VectorObj<TNum> vec(size, value);
    return vec;
}

// Test Fixture for GradientDescent
class GradientDescentTest : public ::testing::Test {
protected:
    int size_ = 100;
    DenseObj<double> P = CreateIdentityMatrix<double>(size_);
    DenseObj<double> A = CreateIdentityMatrix<double>(size_);
    VectorObj<double> b = CreateVector<double>(size_, 1.0);
    VectorObj<double> x = CreateVector<double>(size_, 0.0);
    std::unique_ptr<GradientDescent<double, DenseObj<double>, VectorObj<double>>> gradientDescent;

    void SetUp() override {
        gradientDescent = std::make_unique<GradientDescent<double, DenseObj<double>, VectorObj<double>>>(P, A, b, 100);
    }
};

// Test the constructor of GradientDescent.
TEST_F(GradientDescentTest, Constructor) {
    EXPECT_EQ(gradientDescent->getMaxIter(), 100);
}

// Test the calculation of gradient.
TEST_F(GradientDescentTest, CalGrad) {
    VectorObj<double> expectedGradient = b - (A * x);
    VectorObj<double> actualGradient = gradientDescent->calGrad(x);
    EXPECT_NEAR((actualGradient - expectedGradient).L2norm(), 0.0, 1e-8);
}

// Test the update step in GradientDescent.
TEST_F(GradientDescentTest, Update) {
    VectorObj<double> expectedX = CreateVector<double>(size_, 1.0);
    x = CreateVector<double>(size_, 0.0);
    gradientDescent->solve(x);
    EXPECT_NEAR((x - expectedX).L2norm(), 0.0, 1e-8);
}

// Test Fixture for StaticIterMethod
class StaticIterMethodTest : public ::testing::Test {
protected:
    int size_ = 100;
    DenseObj<double> A = CreateIdentityMatrix<double>(size_);
    VectorObj<double> b = CreateVector<double>(size_, 1.0);
    VectorObj<double> x = CreateVector<double>(size_, 0.0);
    std::unique_ptr<StaticIterMethod<double, DenseObj<double>, VectorObj<double>>> staticIterMethod;

    void SetUp() override {
        staticIterMethod = std::make_unique<StaticIterMethod<double, DenseObj<double>, VectorObj<double>>>(A, A, b, 100);
    }
};

// Test the constructor of StaticIterMethod.
TEST_F(StaticIterMethodTest, Constructor) {
    EXPECT_EQ(staticIterMethod->getMaxIter(), 100);
}

// Test the Substitution method of StaticIterMethod.
TEST_F(StaticIterMethodTest, Substitution) {
    DenseObj<double> L = CreateIdentityMatrix<double>(size_);
    VectorObj<double> result = staticIterMethod->Substitution(b, L, true);
    EXPECT_NEAR((result - b).L2norm(), 0.0, 1e-8);  // L is identity, so result should be b
}

// Test the update step in StaticIterMethod.
TEST_F(StaticIterMethodTest, Update) {
    VectorObj<double> expectedX = CreateVector<double>(size_, 1.0);
    x = CreateVector<double>(size_, 0.0);
    staticIterMethod->solve(x);
    EXPECT_NEAR((x - expectedX).L2norm(), 0.0, 1e-8);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#include <gtest/gtest.h>
#include <cstring> // for memset
#include "IterSolver.hpp"

// Test fixture for IterSolverBase
class GradientDesentTest : public ::testing::Test {
protected:
    MatrixObj<double> P;
    MatrixObj<double> A;
    VectorObj<double> b;
    VectorObj<double> x;
    int max_iter;
    double* I; // Pointer to the dynamically allocated identity matrix

    void SetUp() override {
        const int n = 4; // Number of rows
        const int m = 4; // Number of columns (for a square matrix, n == m)
        I = new double[n*m];
        for (int i = 0; i < n*m; ++i) {
            I[i] = 0.0;
        }
        for (int i = 0; i < n; ++i) {
            I[i*m + i] = 1.0;
        }
        P = MatrixObj<double>(I, n, m);
        A = MatrixObj<double>(I, n, m);

        double* Ivec = new double[n];
        memset(Ivec, 0, n * sizeof(double));
        Ivec[0] = 1.0;
        b = VectorObj<double>(Ivec, n);
        Ivec[0] = 0;
        x = VectorObj<double>(Ivec, n);
        max_iter = 1000;
    }

    void TearDown() override {
        delete[] I; // Clean up the dynamically allocated identity matrix
    }
};

// Test case for the constructor
TEST_F(GradientDesentTest, Constructor) {
    GradientDesent<double> solver(P, A, b, x, max_iter);
    EXPECT_EQ(solver.getIter(), max_iter);
    EXPECT_NEAR(solver.Ans(), x[0], 1e-6); // Compare using a tolerance
}

// Test case for the Update method
TEST_F(GradientDesentTest, Update) {
    GradientDesent<double> solver(P, A, b, x, max_iter);
    solver.alp = 0.1; // Set a learning rate
    // Calculate the expected gradient and update

    MatrixObj<double> xt(x);
	MatrixObj<double> gradt;
    gradt = b - A * xt;
	VectorObj<double> expected_grad = gradt.get_Col(0);

    solver.grad = expected_grad;
    solver.Update();
    // Check if the update is correct
	expected_grad *= solver.alp;
    VectorObj<double> expected_x = x + expected_grad;
    for (int i = 0; i < x.get_row(); ++i) {
        EXPECT_NEAR(solver.Ans(), expected_x[i], 1e-6); // Compare using a tolerance
    }
}

// Test case for the getIter method
TEST_F(GradientDesentTest, GetIter) {
    GradientDesent<double> solver(P, A, b, x, max_iter);
    EXPECT_EQ(solver.getIter(), max_iter);
}

// Test case for the Ans method
TEST_F(GradientDesentTest, Ans) {
    GradientDesent<double> solver(P, A, b, x, max_iter);
    EXPECT_NEAR(solver.Ans(), x[0], 1e-6); // Compare using a tolerance
}

// Test case for the calGrad method in GradientDesent
TEST_F(GradientDesentTest, CalGrad) {
    GradientDesent<double> gd(P, A, b, x, max_iter);
    gd.calGrad();
    // Check if the gradient calculation is correct

    MatrixObj<double> xt(x);
	MatrixObj<double> gradt;
    gradt = b - A * xt;
	VectorObj<double> expected_grad = gradt.get_Col(0);

	for (int i=0; i< b.get_row(); i++){
		EXPECT_NEAR(gd.grad[i], expected_grad[i], 1e-6); // Compare using a tolerance
	}
}

// Test case for the calAlp method in GradientDesent
TEST_F(GradientDesentTest, CalAlp) {
    GradientDesent<double> gd(P, A, b, x, max_iter);
    double hype = 0.1; // Assuming 0.1 is a hyperparameter value
    EXPECT_DOUBLE_EQ(gd.calAlp(hype), hype);
}

// Test case for the Update method in GradientDesent
TEST_F(GradientDesentTest, Update) {
    GradientDesent<double> gd(P, A, b, x, max_iter);
    gd.alp = 0.1; // Set a learning rate
    gd.calGrad();
    gd.Update();
    // Check if the update is correct
	gd.grad *= gd.alp;
    VectorObj<double> expected_x = x + gd.grad;
    for (int i = 0; i < b.get_row(); ++i) {
        ASSERT_NEAR(gd.Ans(), expected_x[i], 1e-6); // Compare using a tolerance
    }
}

// Main function to run all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
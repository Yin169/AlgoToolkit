#include <gtest/gtest.h>
#include "ConjugateGradient.hpp"
#include "TrustRegion.hpp"
#include "SparseObj.hpp"
#include "VectorObj.hpp"

// Test fixture for Conjugate Gradient
class ConjugateGradientTest : public ::testing::Test {
protected:

    ConjugateGradientTest() = default;

    void SetUp() override {
        // Define a simple 3x3 symmetric positive-definite (SPD) matrix
        matrix = SparseMatrixCSC<double>(3, 3);
        matrix.addValue(0, 0, 4);
        matrix.addValue(1, 0, -1);
        matrix.addValue(0, 1, -1);
        matrix.addValue(1, 1, 4);
        matrix.addValue(2, 1, -1);
        matrix.addValue(1, 2, -1);
        matrix.addValue(2, 2, 4);
        matrix.finalize();

        rhs = VectorObj<double>(3);
        rhs[0] = 1;
        rhs[1] = 5;
        rhs[2] = 0;

        // Correct expected solution
        expected_solution = VectorObj<double>(3);
        expected_solution[0] = 0.625;
        expected_solution[1] = 1.5;
        expected_solution[2] = 0.375;

        // matrix  = SparseMatrixCSC<double>(4, 4);
        // matrix.addValue(0, 0, 4.0);
        // matrix.addValue(1, 1, 3.0);
        // matrix.addValue(2, 2, 2.0);
        // matrix.addValue(3, 3, 1.0);
        // matrix.finalize();

        for(int i=0; i<3; i++){
            for(int j=0; j<3; j++){
                std::cout << matrix(i, j) << " ";
            } 
            std::cout << std::endl;  
        }

        // rhs = VectorObj<double>(4);
        // rhs[0] = 8.0;
        // rhs[1] = 9.0;
        // rhs[2] = 4.0;
        // rhs[3] = 1.0;

        // expected_solution = VectorObj<double>(4);
        // expected_solution[0] = 2.0;
        // expected_solution[1] = 3.0;
        // expected_solution[2] = 2.0;
        // expected_solution[3] = 1.0;
    }

    SparseMatrixCSC<double> matrix;
    VectorObj<double> rhs;
    VectorObj<double> expected_solution;
};

// Test for convergence on a simple SPD matrix
TEST_F(ConjugateGradientTest, SolvesSPDMatrix) {
    ConjugateGrad<double, SparseMatrixCSC<double>, VectorObj<double>> cg_solver(matrix, rhs, 1000, 1e-6);

    VectorObj<double> solution(rhs.size(), 0.0);
    cg_solver.solve(solution);

    for (int i = 0; i < solution.size(); ++i) {
        EXPECT_NEAR(solution[i], expected_solution[i], 1e-5) << "Mismatch at index " << i;
    }
}

// Test for zero RHS
TEST_F(ConjugateGradientTest, ZeroRHS) {
    VectorObj<double> zero_rhs(rhs.size(), 0.0);
    VectorObj<double> solution(rhs.size(), 0.0);

    ConjugateGrad<double, SparseMatrixCSC<double>, VectorObj<double>> cg_solver(matrix, zero_rhs, 1000, 1e-6);
    cg_solver.solve(solution);

    for (int i = 0;  i < solution.size(); ++i) {
        EXPECT_NEAR(solution[i], 0.0, 1e-12);
    }
}

// Test for a single iteration (useful for debugging)
TEST_F(ConjugateGradientTest, SingleIteration) {
    ConjugateGrad<double, SparseMatrixCSC<double>, VectorObj<double>> cg_solver(matrix, rhs, 1, 1e-6);

    VectorObj<double> solution(rhs.size(), 0.0);
    cg_solver.solve(solution);

    // Ensure the solution is not converged but has changed
    EXPECT_GT(std::abs(rhs[0] - matrix(0, 0) * solution[0]), 1e-6);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#include <gtest/gtest.h>
#include "MultiGrid.hpp"
#include "SparseObj.hpp" // For SparseMatrixCSC
#include "IterSolver.hpp" // For iterative solvers like Jacobi
#include <vector>

// template <typename TNum>
// using SparseMatrixCSC = SparseMatrixCSC<TNum>;

// Test fixture for MultiGrid
class MultiGridTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Define a simple 3x3 Laplacian matrix for testing
        matrix = SparseMatrixCSC<double>(3, 3);
        matrix.addValue(0, 0, 4);
        matrix.addValue(0, 1, -1);
        matrix.addValue(1, 0, -1);
        matrix.addValue(1, 1, 4);
        matrix.addValue(1, 2, -1);
        matrix.addValue(2, 1, -1);
        matrix.addValue(2, 2, 4);
        matrix.finalize();

        rhs = VectorObj<double>(3);
        rhs[0] = 1;
        rhs[1] = 2;
        rhs[2] = 3;
        expected_solution = VectorObj<double>(3);
        expected_solution[0] = 1.0 / 16;
        expected_solution[0] = 5.0 / 16;
        expected_solution[0] = 3.0 / 8;   
    }

    SparseMatrixCSC<double> matrix;
    SparseMatrixCSC<double> P;
    VectorObj<double> rhs;
    VectorObj<double> expected_solution;
};

// Test MultiGrid solver for convergence
TEST_F(MultiGridTest, SolvesSimpleLaplacian) {
    AlgebraicMultiGrid<double, VectorObj<double>> mg_solver;

    VectorObj<double> zero_rhs(rhs.size(), 0.0);
    VectorObj<double> solution(rhs.size(), 0.0);
    mg_solver.amgVCycle(matrix, zero_rhs, solution, 5, 100, 1.5);

    for (size_t i = 0; i < solution.size(); ++i) {
        EXPECT_NEAR(solution[i], expected_solution[i], 1e-5) << "Mismatch at index " << i;
    }
}

// Test for zero RHS (should return a zero vector)
TEST_F(MultiGridTest, ZeroRHS) {
    VectorObj<double> zero_rhs(rhs.size(), 0.0);
    VectorObj<double> solution(rhs.size(), 0.0);

    AlgebraicMultiGrid<double, VectorObj<double>> mg_solver;

    mg_solver.amgVCycle(matrix, zero_rhs, solution, 5, 100, 1.5);

    for (int i = 0;  i < solution.size(); ++i) {
        EXPECT_NEAR(solution[i], 0.0, 1e-12);
    }
}

// Test for a single iteration (useful for debugging intermediate steps)
TEST_F(MultiGridTest, SingleIteration) {
    AlgebraicMultiGrid<double, VectorObj<double>> mg_solver;

    VectorObj<double> zero_rhs(rhs.size(), 0.0);
    VectorObj<double> solution(rhs.size(), 0.0);
    mg_solver.amgVCycle(matrix, zero_rhs, solution, 5, 100, 1.5);

    // Check that the solution is not fully converged but has improved
    EXPECT_GT(std::abs(rhs[0] - matrix(0, 0) * solution[0]), 1e-6);
}

// Test for handling non-square matrices (should throw an exception)
TEST_F(MultiGridTest, NonSquareMatrix) {
    SparseMatrixCSC<double> non_square_matrix(2, 3);
    non_square_matrix.addValue(0, 0, 1);
    non_square_matrix.addValue(1, 2, 1);
    non_square_matrix.finalize();

    VectorObj<double> rhs_non_square(2);
    rhs_non_square[0] = 1;
    rhs_non_square[1] = 2;
    VectorObj<double> solution(rhs_non_square.size(), 0.0);

    AlgebraicMultiGrid<double, VectorObj<double>> mg_solver;

    EXPECT_THROW(mg_solver.amgVCycle(non_square_matrix, rhs_non_square, solution, 5, 100, 1.5), std::invalid_argument);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
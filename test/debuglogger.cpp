#include <gtest/gtest.h>
#include "IterSolver.hpp"
#include "ConjugateGradient.hpp"

TEST(Solver, DebuggingIntermediateSteps) {
    SparseMatrixCSC<double> A(4, 4);
    A.addValue(0, 0, 4.0);
    A.addValue(1, 1, 3.0);
    A.addValue(2, 2, 2.0);
    A.addValue(3, 3, 1.0);
    A.finalize();

    std::cout << " ----finish---- " << std::endl;
    VectorObj<double> b(4);
    b[0] = 8.0;
    b[1] = 9.0;
    b[2] = 4.0;
    b[3] = 1.0;

    VectorObj<double> x(4, 0.0); // Initial guess

    // GradientDescent<double, SparseMatrixCSC<double>, VectorObj<double>> solver(A, b, 100, 1e-6);
    // StaticIterMethod<double, SparseMatrixCSC<double>, VectorObj<double>> solver(A, b, 1000, 0.5);
    ConjugateGrad<double, SparseMatrixCSC<double>, VectorObj<double>> solver(A, b, 10, 1e-6);

    // Hook to capture intermediate steps
    struct DebugLogger {
        static void log(int iter, const VectorObj<double>& x, double residual) {
            std::cout << "Iteration " << iter << ": Residual = " << residual << ", Solution = [";
            for (int i = 0; i < x.size(); ++i) std::cout << x[i] << (i + 1 < x.size() ? ", " : "");
            std::cout << "]\n";
        }
    };

    // Modified solve method with debugging
    // VectorObj<double> r = b - (A * x); // Initial residual
    // double residualNorm = r.L2norm();
    // for (int iter = 0; iter < 100 && residualNorm > 1e-6; ++iter) {
    //     VectorObj<double> Ar = A * r;
    //     double alpha = (r * r) / (r * Ar);
    //     x = x + r * alpha;
    //     r = r - Ar * alpha;
    //     residualNorm = r.L2norm();

    //     DebugLogger::log(iter, x, residualNorm); // Log step

    //     if ((x - (x - r * alpha)).L2norm() < 1e-6) break; // Early stopping
    // }

	solver.solve(x);

    EXPECT_NEAR(x[0], 2.0, 1e-6);
    EXPECT_NEAR(x[1], 3.0, 1e-6);
    EXPECT_NEAR(x[2], 2.0, 1e-6);
    EXPECT_NEAR(x[3], 1.0, 1e-6);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

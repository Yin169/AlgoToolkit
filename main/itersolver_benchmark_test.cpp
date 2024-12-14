#include "../src/LinearAlgebra/Solver/IterSolver.hpp"
#include <benchmark/benchmark.h>

// Helper function to create a random matrix and vector for benchmarking.
template<typename TNum>
void CreateRandomMatrixAndVector(MatrixObj<TNum> &A, VectorObj<TNum> &b, VectorObj<TNum> &x, int size) {
    A = MatrixObj<TNum>(size, size);
    b = VectorObj<TNum>(size);
    x = VectorObj<TNum>(size);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            A[i * size + j] = static_cast<TNum>(std::rand() % 100) / 50.0 + 1.0; // Random value between 1 and 2
        }
        b[i] = static_cast<TNum>(std::rand() % 100) / 50.0 + 1.0; // Random value between 1 and 2
        x[i] = 0; // Initialize x to zero for the solver
    }
}

// Benchmark for the GradientDesent solver.
template<typename TNum>
static void BM_GradientDesent(benchmark::State& state) {
    int size = state.range(0);
    MatrixObj<TNum> A;
    VectorObj<TNum> b;
    VectorObj<TNum> x;

    // Create a random matrix and vector before the benchmark runs.
    CreateRandomMatrixAndVector(A, b, x, size);

    // Create a GradientDesent solver instance.
    GradientDesent<TNum> solver(A, A, b, size);

    for (auto _ : state) {
        solver.callUpdate(x);
    }
}

// Benchmark for the Jacobi solver.
template<typename TNum>
static void BM_Jacobi(benchmark::State& state) {
    int size = state.range(0);
    MatrixObj<TNum> A;
    VectorObj<TNum> b;
    VectorObj<TNum> x;

    // Create a random matrix and vector before the benchmark runs.
    CreateRandomMatrixAndVector(A, b, x, size);

    // Create a Jacobi solver instance.
    Jacobi<TNum> solver(A, A, b, size);

    for (auto _ : state) {
        solver.callUpdate(x);
    }
}

// Register the benchmarks.
BENCHMARK_TEMPLATE(BM_GradientDesent, double)->Range(8, 8<<6); // You can adjust the range as needed.
BENCHMARK_TEMPLATE(BM_Jacobi, double)->Range(8, 8<<6); // You can adjust the range as needed.

// Main function to run the benchmarks.
BENCHMARK_MAIN();
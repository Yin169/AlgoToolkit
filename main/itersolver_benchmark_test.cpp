#include "IterSolver.hpp"
#include <benchmark/benchmark.h>

// Helper function to create a random matrix and vector for benchmarking.
template<typename TNum>
void CreateRandomMatrixAndVector(MatrixObj<TNum>& A, VectorObj<TNum>& b, VectorObj<TNum>& x, int size) {
    A = MatrixObj<TNum>(size, size);
    b = VectorObj<TNum>(size);
    x = VectorObj<TNum>(size, 0.0);  // Initialize x to zero directly

    // Fill matrix A and vector b with random values between 1 and 2.
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            A[i * size + j] = static_cast<TNum>(std::rand() % 100) / 50.0 + 1.0;
        }
        b[i] = static_cast<TNum>(std::rand() % 100) / 50.0 + 1.0;
    }
}

// Benchmark for the GradientDesent solver.
template<typename TNum>
static void BM_GradientDesent(benchmark::State& state) {
    int size = state.range(0);
    MatrixObj<TNum> A;
    VectorObj<TNum> b, x;

    // Create a random matrix and vector before the benchmark runs.
    CreateRandomMatrixAndVector(A, b, x, size);

    // Create a GradientDesent solver instance.
    GradientDescent<TNum> solver(A, A, b, size);

    for (auto _ : state) {
        solver.callUpdate(x);  // Perform the update step in the solver
    }
}

// Benchmark for the StaticIterMethod solver.
template<typename TNum>
static void BM_StaticIterMethod(benchmark::State& state) {
    int size = state.range(0);
    MatrixObj<TNum> A;
    VectorObj<TNum> b, x;

    // Create a random matrix and vector before the benchmark runs.
    CreateRandomMatrixAndVector(A, b, x, size);

    // Create a StaticIterMethod solver instance.
    StaticIterMethod<TNum> solver(A, A, b, size);

    for (auto _ : state) {
        solver.callUpdate(x);  // Perform the update step in the solver
    }
}

// Register the benchmarks with appropriate range.
BENCHMARK_TEMPLATE(BM_GradientDesent, double)->Range(8, 8 << 6);  // Benchmark for sizes 8 to 8^6
BENCHMARK_TEMPLATE(BM_StaticIterMethod, double)->Range(8, 8 << 6);  // Benchmark for sizes 8 to 8^6

// Main function to run the benchmarks.
BENCHMARK_MAIN();

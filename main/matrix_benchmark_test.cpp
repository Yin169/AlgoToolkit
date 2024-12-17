#include <benchmark/benchmark.h>
#include "MatrixObj.hpp"
#include "VectorObj.hpp"

// Benchmark for matrix creation
static void BM_MatrixCreation(benchmark::State& state) {
    int rows = state.range(0);
    int cols = state.range(1);
    for (auto _ : state) {
        MatrixObj<int> matrix(rows, cols);
        benchmark::DoNotOptimize(matrix);
    }
}
// Benchmark different sizes for matrix creation
BENCHMARK(BM_MatrixCreation)
    ->ArgPair(100, 100)
    ->ArgPair(1024, 1024)
    ->ArgPair(4096, 4096);

// Benchmark for matrix addition
static void BM_MatrixAddition(benchmark::State& state) {
    int rows = state.range(0);
    int cols = state.range(1);

    // Initialize matrices
    MatrixObj<int> matrix1(rows, cols);
    MatrixObj<int> matrix2(rows, cols);
    for (int i = 0; i < matrix1.size(); ++i) {
        matrix1[i] = i;
        matrix2[i] = i + 1;
    }

    for (auto _ : state) {
        MatrixObj<int> result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
// Benchmark different sizes for matrix addition
BENCHMARK(BM_MatrixAddition)
    ->ArgPair(100, 100)
    ->ArgPair(1024, 1024)
    ->ArgPair(4096, 4096);

// Benchmark for matrix multiplication
static void BM_MatrixMultiplication(benchmark::State& state) {
    int rows_a = state.range(0);
    int cols_a = state.range(1);
    int cols_b = state.range(2);

    // Initialize matrices
    MatrixObj<double> matrix1(rows_a, cols_a);
    MatrixObj<double> matrix2(cols_a, cols_b);
    for (int i = 0; i < matrix1.size(); ++i) {
        matrix1[i] = static_cast<double>(i);
    }
    for (int i = 0; i < matrix2.size(); ++i) {
        matrix2[i] = static_cast<double>(i + 1);
    }

    for (auto _ : state) {
        MatrixObj<double> result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
// Benchmark different sizes for matrix multiplication
BENCHMARK(BM_MatrixMultiplication)
    ->Args({100, 100, 100})
    ->Args({1024, 1024, 1024})
    ->Args({4096, 4096, 4096});

// Benchmark entry point
BENCHMARK_MAIN();

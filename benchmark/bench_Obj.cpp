#include <benchmark/benchmark.h>
#include "DenseObj.hpp"
#include "VectorObj.hpp"
#include "SparseObj.hpp"

// Benchmark for DenseObj matrix multiplication
static void BM_DenseMatrixMultiplication(benchmark::State& state) {
    int n = state.range(0);
    DenseObj<double> A(n, n);
    DenseObj<double> B(n, n);

    // Initialize matrices with some values
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A(i, j) = i + j;
            B(i, j) = i - j;
        }
    }

    for (auto _ : state) {
        DenseObj<double> C = A * B;
        benchmark::DoNotOptimize(C);
    }
}
BENCHMARK(BM_DenseMatrixMultiplication)->Range(8, 1 << 11);

// Benchmark for VectorObj dot product
static void BM_VectorDotProduct(benchmark::State& state) {
    int n = state.range(0);
    VectorObj<double> v1(n);
    VectorObj<double> v2(n);

    // Initialize vectors with some values
    for (int i = 0; i < n; ++i) {
        v1[i] = i;
        v2[i] = i * 2;
    }

    for (auto _ : state) {
        double result = v1 * v2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorDotProduct)->Range(8, 1 << 11);

// Benchmark for SparseMatrixCSC matrix-vector multiplication
static void BM_SparseMatrixVectorMultiplication(benchmark::State& state) {
    int n = state.range(0);
    SparseMatrixCSC<double> A(n, n);
    VectorObj<double> v(n);

    // Initialize sparse matrix and vector with some values
    for (int i = 0; i < n; ++i) {
        A.addValue(i, i, i + 1); // Diagonal matrix
        v[i] = i;
    }
    A.finalize();

    for (auto _ : state) {
        VectorObj<double> result = A * v;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_SparseMatrixVectorMultiplication)->Range(8, 1 << 11);

// Benchmark for SparseMatrixCSC matrix multiplication
static void BM_SparseMatrixMultiplication(benchmark::State& state) {
    int n = state.range(0);
    SparseMatrixCSC<double> A(n, n);
    SparseMatrixCSC<double> B(n, n);

    // Initialize sparse matrices with some values
    for (int i = 0; i < n; ++i) {
        A.addValue(i, i, i + 1); // Diagonal matrix
        B.addValue(i, i, i + 1); // Diagonal matrix
    }
    A.finalize();
    B.finalize();

    for (auto _ : state) {
        SparseMatrixCSC<double> C = A * B;
        benchmark::DoNotOptimize(C);
    }
}
BENCHMARK(BM_SparseMatrixMultiplication)->Range(8, 1 << 11);

BENCHMARK_MAIN();
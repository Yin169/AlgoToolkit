#include <benchmark/benchmark.h>
#include "MatrixObj.hpp"
#include "VectorObj.hpp"

void BM_MatrixCreation(benchmark::State& state) {
    for (auto _ : state) {
        MatrixObj<int> matrix(state.range(0), state.range(1));
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_MatrixCreation)->ArgPair(100, 100)->ArgPair(4096, 4096);

void BM_MatrixAddition(benchmark::State& state) {
    MatrixObj<int> matrix1(state.range(0), state.range(1));
    MatrixObj<int> matrix2(state.range(0), state.range(1));
    for (auto _ : state) {
        MatrixObj<int> result = matrix1 + matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixAddition)->ArgPair(100, 100)->ArgPair(4096, 4096);

void BM_MatrixMultiplication(benchmark::State& state) {
    MatrixObj<double> matrix1(state.range(0), state.range(1));
    MatrixObj<double> matrix2(state.range(1), state.range(2));
    for (auto _ : state) {
        MatrixObj<double> result = matrix1 * matrix2;
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_MatrixMultiplication)->Args({4096, 4096, 4096});

BENCHMARK_MAIN();
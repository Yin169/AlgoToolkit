#include <cblas.h>
#include <stdio.h>

int main() {
    const CBLAS_LAYOUT Layout = CblasRowMajor;
    const CBLAS_TRANSPOSE TransA = CblasNoTrans;
    const CBLAS_TRANSPOSE TransB = CblasNoTrans;
    const int M = 2 ;
    const int N = 2;
    const int K = 2;
    const double alpha = 1.0;
    const double beta = 0.0;
    double A[4] = {1, 2, 3, 4};
    double B[4] = {5, 6, 7, 8};
    double C[4] = {0, 0, 0, 0};
    const int lda = 2;
    const int ldb = 2;
    const int ldc = 2;

    cblas_dgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%lf ", C[i * ldc + j]);
        }
        printf("\n");
    }

    return 0;
}